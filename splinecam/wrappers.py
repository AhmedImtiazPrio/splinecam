import torch
from .graph import _batched_gpu_op
from . import utils
import warnings
from itertools import combinations
import numpy as np

from .global_config import SUPPORTED_MODULES, SUPPORTED_ACT

class adapter(torch.jit.ScriptModule):
    """
    Adapter head for a network that maps a 2D region to the ROI of the network input.

    Attributes
    ----------
    basis_vec : Tensor
        Tensor with two orthogonal basis vectors that maps R2->RD where D is the input dimension
    name : str
        name of module
    device : {'cuda','cpu'}
        device to store parameters in    
    out_shape,in_shape : int
        RD,R2
    Abw : Tensor
        contains the basis vectors as hyperplanes

    Methods
    -------
    forward(self,x):
        maps from R2->RD
    to_input(self,x):
        maps from RD->R2
    get_Abw(self):
        returns Abw
    
    """
    
    def __init__(self,T,name='adapter',
                 device='cuda', dtype=torch.float32):
        """
        T : projection mat.
        """
        
        super(adapter, self).__init__()
        
        self.device = device
        self.dtype = dtype
        
        self.T = T.to(self.device).type(self.dtype)
            
        self.output_shape = T.shape[0]
        self.input_shape = T.shape[1]-1
        
#         self.Abw = torch.hstack([self.basis_vec,self.offset_vec])
        self.Abw = self.T
    
    @torch.no_grad()
    def forward(self,x):
        return (self.Abw[...,:-1] @ x.T + self.Abw[...,-1:]).T
    
    @torch.no_grad()
    def to_input(self,x):
#         return (self.basis_vec.T @ x.T).T
        return (self.Abw[...,:-1].T @ (x.T - self.Abw[...,-1:])).T
    
    @torch.no_grad()
    def get_weights(self,dtype=torch.float64):      
        return self.Abw.type(dtype)
    
    @torch.no_grad()
    def get_activation_pattern(self,x):
        return torch.ones(x.shape[0],self.output_shape,
                          device=x.device, dtype=x.dtype) ##self.dtype issue due to jit
    @torch.no_grad()
    def get_intersection_pattern(self,x):
        return torch.ones(x.shape[0],self.output_shape,
                          device=x.device, dtype=x.dtype)

class linear(torch.nn.Module):
    """
    Wrapper for a torch.nn.Linear layer with batchnorm and activation function
    Module forward is in sequence Linear->BN->Activation

    Attributes
    ----------
    
    has_act,has_bn : bool:
        whether layer contains batchnorm or activation function
    act,bn : torch.nn.Module
        Batchnorm and activation torch module
    name : str
        name of module
    device : {'cuda','cpu'}
        device to store parameters in    
    
    Methods
    -------
    TODO: complete docstring
    
    """
    
    def __init__(self,linear_layer, is_classifier=False,
                 act_layer=None,bn_layer=None,name='linear_layer',
                 device='cuda', dtype=torch.float32):
        '''
        Wrapper for linear layer with batchnorm and/or ReLU activation
        Follows order Linear->BN->Activation
        '''
        super(linear, self).__init__()
        
        self.device = device
        self.dtype = dtype
        self.name = name
        self.is_classifier = is_classifier
        
        # check if layer has act and bn
        self.has_act = act_layer is not None
        self.has_bn = bn_layer is not None
        
        self.layer = linear_layer.to(self.device).type(self.dtype)
        
        self.input_shape = torch.tensor(self.layer.get_parameter('weight').shape[1])
        self.output_shape = torch.tensor(self.layer.get_parameter('weight').shape[0])
        
        if self.has_act:
            self.add_act(act_layer)
        else:
            self.act = lambda x:x
        
        if self.has_bn:
            self.add_bn(bn_layer)
        else:
            self.bn = lambda x:x
            
        if is_classifier:
            self.to_decision_boundary()
        
    @torch.no_grad() #TODO: Make these methods jit ignore
    def add_bn(self,bn_layer):
        
        self.has_bn = True
        
        self.bn = bn_layer.to(self.device).type(self.dtype)
        self.bn_rmean = self.bn.__dict__['_buffers']['running_mean']
        self.bn_rvar = self.bn.__dict__['_buffers']['running_var']
        self.bn_gamma = self.bn.__dict__['_parameters']['weight']
        self.bn_beta = self.bn.__dict__['_parameters']['bias']
        self.bn_eps = self.bn.__dict__['eps']
        
        
    
    @torch.no_grad() #TODO: Make these methods jit ignore
    def add_act(self,act_layer):    
        
        self.has_act = True
        
        self.act = act_layer.to(self.device).type(self.dtype)
        
        if type(self.act) == torch.nn.modules.activation.LeakyReLU:
            self.act_name = 'lrelu'
            self.act_nslope = self.act.__dict__['negative_slope']
            
        elif type(self.act) == torch.nn.modules.activation.ReLU:
            self.act_name = 'relu'
            
        else:
            raise NotImplementedError('Activation func not supported')
            
            
    @torch.no_grad()
    def to_decision_boundary(self,between_class=None):
        
        self.between_class = between_class
        
        if hasattr(self,'Ab'):
            del self.Ab ## to reset Ab
        
        W = self.get_weights()

        if between_class is None:

            n_hyps = W.shape[0]
            
            hyp_group_idx = torch.from_numpy(
                np.asarray(list(combinations(range(n_hyps),2)))
            ).type(torch.int64)

            self.Ab = W[hyp_group_idx[:,0]] - W[hyp_group_idx[:,1]]

        if between_class is not None:

            self.Ab = (W[between_class[0]] - W[between_class[1]])[None,...]

        self.is_classifier = True
        self.output_shape = torch.tensor(self.Ab.shape[0])
        
        return self.is_classifier
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_weights(self,row_idx=None,dtype=torch.float64):
        '''
        Returns Ab = [Aw|bw]
        if row_idx specified, only returns those hyps/rows
        '''
        
        if not hasattr(self,'Ab'):
            
            if self.has_bn:
                
                c = self.bn_gamma/(self.bn_rvar + self.bn_eps)**.5
                b = self.bn_beta - self.bn_rmean*c
                
                self.Ab = torch.hstack((
                    c[...,None]*self.layer.get_parameter('weight'),
                    (c*self.layer.get_parameter('bias') + b)[...,None]
                    )
                )
                
            else:
                
                self.Ab = torch.hstack((
                    self.layer.get_parameter('weight'),
                    self.layer.get_parameter('bias')[...,None]
                    )
                )
        
        if row_idx is None:
            return self.Ab.type(dtype).to(self.device)
        else:
            return self.Ab[row_idx].to(self.device)
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_activation_pattern(self,x):
        '''
        Get VQ code
        '''
        
        pre_act = self.layer.forward(x)
        
        if self.has_bn:
            pre_act = self.bn.forward(pre_act)

        if not self.has_act:
            q = torch.ones_like(pre_act)
        elif self.act_name == 'lrelu':
            q = (pre_act>0)*1. + (pre_act<=0)*self.act_nslope
        elif self.act_name == 'relu':
            q = (pre_act>0)*1.
            
        return q
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_intersection_pattern(self,x):
        '''
        Get sign of x wrt layer hyperplanes
        '''
        
        if not self.is_classifier:
            
            pre_act = self.layer.forward(x)

            if self.has_bn:
                pre_act = self.bn.forward(pre_act)
        
        elif self.is_classifier:
            
            pre_act = (self.Ab[...,:-1] @ x.T + self.Ab[...,-1:]).T
        
        return (pre_act>0)*1.
    
#     @torch.no_grad()
#     @torch.jit.script_method TODO: Make these methods jittable
    @torch.no_grad()
    def forward(self,x):
        '''
        Follows order Linear->BN->activation by default
        '''
        x = self.layer.forward(x)
        
        if self.has_bn:
            x = self.bn.forward(x)
        if self.has_act:
            x = self.act.forward(x)
        
        return x
    
class model_wrapper(object):
    """
    TODO: Complete docstring
    """

    
    @torch.no_grad()
    def __init__(self, model, input_shape=None, is_classifier=False,
                 custom_layers = [], custom_activations = [], 
                 device='cuda',as_sequential=True,T=None, dtype=torch.float32):
        
        self.device = device
        self.dtype = dtype
        self.as_sequential = as_sequential
        self.T = T
        self.input_shape = input_shape
        self.is_classifier = is_classifier
        self.custom_layers = custom_layers
        self.custom_activations = custom_activations
        
        if model.training:
            raise ValueError('Model should always be supplied in eval mode. Please do `model.eval()` before wrapping')
        
        model_dict = model.__dict__
        layers = [v for v in model_dict['_modules'].values()]
        layer_names = [k for k in model_dict['_modules'].keys()]
        
        self.layers = []
        
        if T is not None:
            self.layers.append(adapter(T=self.T, dtype=self.dtype, device=self.device))
            
        self.layers += self.compose_layers(layers,layer_names)
                
        if self.as_sequential:
            self.layers = torch.nn.modules.Sequential(*self.layers)
            
    @torch.no_grad()        
    def compose_layers(self,layers,layer_names):
        
        global SUPPORTED_MODULES, SUPPORTED_ACT
        
        SUPPORTED_MODULES += self.custom_layers
        SUPPORTED_ACT += self.custom_activations
        
        not_supported = [type(each) for i,each in enumerate(layers) if not type(
            each) in SUPPORTED_MODULES+SUPPORTED_ACT]
        
        if len(not_supported)>0:
            raise NotImplementedError(f'Model contains modules {not_supported} which are currently not supported')
        
        current_shape = self.input_shape
        
        new_layers = []
        
        for layer,name in zip(layers,layer_names):
            
            print(f'Wrapping layer {name}...')
            
            if type(layer) == torch.nn.modules.linear.Linear:
                new_layers.append(linear(layer,device=self.device,dtype=self.dtype))

            elif type(layer) == torch.nn.modules.Conv2d:
                
                if current_shape is None:
                    raise ValueError('Please specify model input shape for conv networks')
                
                new_layers.append(conv2d(layer,input_shape=current_shape,
                                         device=self.device,dtype=self.dtype)
                                 )
                current_shape = new_layers[-1].output_shape
            
            elif type(layer) == torch.nn.modules.AvgPool2d:
                
                if current_shape is None:
                    raise ValueError('Please specify model input shape for avgpool networks')
                    
                new_layers.append(avgpool2d(layer,input_shape=current_shape,
                                            device=self.device,dtype=self.dtype)
                                 )
                current_shape = new_layers[-1].output_shape
            
            elif type(layer) in self.custom_layers:
                
                if current_shape is None:
                    raise ValueError('Please specify model input shape for avgpool networks')
                    
                
                layer.device = self.device
                layer.dtype = self.dtype
                layer.input_shape = current_shape
                
                new_layers.append(layer)
                
                current_shape = new_layers[-1].output_shape
             
            
            elif type(layer) == torch.nn.modules.Flatten:
                pass ## conv outputs are already flattened via x.reshape(-1)
                
            ## if act add to prev layer
            elif type(layer) in SUPPORTED_ACT:
                new_layers[-1].add_act(layer)
            
            ## if bn add to prev layer
            elif type(layer) == torch.nn.modules.BatchNorm1d or type(layer) == torch.nn.modules.BatchNorm2d:
                new_layers[-1].add_bn(layer)
                
            else:
                pass #pass if dropout, sequential
                
        return new_layers
        
    @torch.no_grad()          
    def verify(self,seed=0,eps=1e-5):
        '''
        verify forward and matmul outputs are identical
        '''
        
        torch.manual_seed(seed)
        if self.T is not None:
            A = torch.randn(1,2, device=self.device).type(self.dtype)
        else:
            A = torch.randn(1,*self.input_shape, device=self.device).type(self.dtype).reshape(1,-1)
        
        if not self.as_sequential:
            fwd_val = torch.nn.modules.Sequential(*self.layers).forward(A)
        else:
            fwd_val = self.layers.forward(A)

        for each_layer in self.layers:
            Ab = each_layer.get_weights()
            W,b = utils.split_sparse_Ab(Ab)
            
            pre_act = (W @ A.T + b).T
            q = each_layer.get_activation_pattern(A)
            A = q*pre_act

        return torch.allclose(A,fwd_val,rtol=0,atol=eps)

class conv2d(torch.nn.Module):
    """
    Wrapper for a torch.nn.Linear layer with batchnorm and activation function
    Module forward is in sequence Linear->BN->Activation

    Attributes
    ----------
    
    has_act,has_bn : bool:
        whether layer contains batchnorm or activation function
    act,bn : torch.nn.Module
        Batchnorm and activation torch module
    name : str
        name of module
    device : {'cuda','cpu'}
        device to store parameters in    
    
    Methods
    -------
    TODO: complete docstring
    
    """
    
    def __init__(self,conv2d_layer,input_shape,
                 act_layer=None,bn_layer=None,custom_activations=None,
                 name='conv2d_layer',
                 device='cuda', dtype=torch.float32):
        '''
        Wrapper for linear layer with batchnorm and/or ReLU activation
        Follows order Linear->BN->Activation
        '''
        super(conv2d, self).__init__()
        
        self.device = device
        self.name = name
        self.dtype = dtype
        self.custom_activations = custom_activations
        
        self.input_shape = torch.tensor(input_shape)
        self.layer = conv2d_layer.to(self.device).type(self.dtype)
        
        #enforcing symmetric
        assert self.layer.stride[0] == self.layer.stride[1]        
        if not self.layer.kernel_size[0] == self.layer.kernel_size[1]:
            warnings.warn(f'Kernel size {self.layer.kernel_size} non symmetric')
            
        if not type(self.layer.padding) == str:
            assert self.layer.padding[0] == self.layer.padding[1]
            assert self.layer.kernel_size[0] - 2*self.layer.padding[0] == 1 ## padding same

        else:
            warnings.warn(f'Padding {self.layer.padding}')
        
        self.n_kernels = self.layer.weight.shape[0]
        self.stride = self.layer.stride[0]
        
        self.output_shape = torch.tensor(
            (self.n_kernels,
             self.input_shape[1]/self.stride,
             self.input_shape[2]/self.stride)
        ).type(torch.int64)
        
        # check if layer has act and bn
        self.has_act = act_layer is not None
        self.has_bn = bn_layer is not None       
        
        if self.has_act:
            self.add_act(act_layer)
        else:
            self.act = lambda x:x
        
        if self.has_bn:
            self.add_bn(bn_layer)
        else:
            self.bn = lambda x:x
            
#         self.Ab = self.prepare_weights()
            
        
    @torch.no_grad() #TODO: Make these methods jit ignore
    def add_bn(self,bn_layer):
        
        self.has_bn = True
        
        self.bn = bn_layer.to(self.device).type(self.dtype)
        self.bn_rmean = self.bn.__dict__['_buffers']['running_mean']
        self.bn_rvar = self.bn.__dict__['_buffers']['running_var']
        self.bn_gamma = self.bn.__dict__['_parameters']['weight']
        self.bn_beta = self.bn.__dict__['_parameters']['bias']
        self.bn_eps = self.bn.__dict__['eps']
        
        self.Ab = self.prepare_weights()
        
    
    @torch.no_grad() #TODO: Make these methods jit ignore
    def add_act(self,act_layer):    
        
        self.has_act = True
        
        self.act = act_layer.to(self.device).type(self.dtype)
        
        if type(self.act) == torch.nn.modules.activation.LeakyReLU:
            self.act_name = 'lrelu'
            self.act_nslope = self.act.__dict__['negative_slope']
            
        elif type(self.act) == torch.nn.modules.activation.ReLU:
            self.act_name = 'relu'
            
        elif type(self.act) in self.custom_activations:
            self.act_name = 'custom'
            raise NotImplementedError('Custom activation support not implemented')
            
        else:
            raise NotImplementedError('Activation func not supported')
            
    @torch.no_grad() #TODO: make jittable
    def prepare_weights(self):
        '''
        Get matrix representation of conv
        '''
    
        identity = torch.eye(
            torch.prod(
                self.input_shape
            ).type(torch.int64), device='cpu', dtype=torch.float32
        ).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2]
                 )
        
        self.layer.type(torch.float32)
        
        if self.has_bn:
            
            c = self.bn_gamma/(self.bn_rvar + self.bn_eps)**.5
            b = self.bn_beta - self.bn_rmean*c
            
            c = c.type(torch.float32)
            b = b.type(torch.float32)
            
        
        if self.has_bn and self.layer.bias is None:
            
            method = lambda x: c[None,:,None,None]*self.layer.forward(x)
        
        elif self.has_bn and self.layer.bias is not None:
        
            method = lambda x: c[None,:,None,None]*(self.layer.forward(x) - self.layer.bias[None,:,None,None])
        
        elif not self.has_bn and self.layer.bias is not None:
            
            method = lambda x: self.layer.forward(x)-self.layer.bias[None,:,None,None]
            
        else:
            method = self.layer.forward
            
        output = _batched_gpu_op(
            method = method, data=identity, batch_size=2048*2, dtype=torch.float32,workers=0,
            out_size=(identity.shape[0],self.output_shape[0],self.output_shape[1],self.output_shape[2]) 
        )
            
#         output = self.layer.forward(identity)
    
        W = output.reshape(-1, torch.prod(self.output_shape).type(torch.int64)).T

        if self.layer.bias is not None:
            
            bias = self.layer.bias.data

            
            if self.has_bn:
                bias = c*self.layer.bias.data + b
                
            b = torch.stack([torch.ones(self.output_shape[1],self.output_shape[2]
                                       ).type(torch.float32).to(self.device) * bi for bi in bias])
            b = b.reshape(torch.prod(self.output_shape).type(torch.int64),1)

        else:

            bias = torch.zeros(self.output_shape[0]).type(torch.float32).to(self.device)
            
            if self.has_bn:
                bias += b
                
            b = torch.stack([torch.ones(self.output_shape[1],self.output_shape[2]
                                       ).type(torch.float32).to(self.device) * bi for bi in bias])
            b = b.reshape(torch.prod(self.output_shape).type(torch.int64),1)
        
        b = b.to(W.device)
        self.layer.type(self.dtype)
        
        try: ## sparse matrix not allowed for very large matrices
            W = torch.hstack([W,b]).to_sparse().cpu().type(self.dtype) 
        except RuntimeError:
            warnings.warn('Abw too big for creating sparse matrix')
            W = torch.hstack([W,b]).cpu().type(self.dtype)
        return W
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_weights(self,row_idx=None,dtype=torch.float64):
        '''
        Returns Ab = [Aw|bw]
        if row_idx specified, only returns those hyps/rows
        defualt stores Ab in cpu and returns gpu during calls
        '''
        
        if not hasattr(self,'Ab'):
            ## get matrix representation of weights
            self.Ab = self.prepare_weights()
                
#         if row_idx is None:
#             return self.Ab.to_dense().to(self.device) 
#         else:
#             return self.Ab.to_dense()[row_idx].to(self.device)
        
        if row_idx is None:
            return self.Ab.type(dtype).to(self.device)
        else:
#             return self.Ab[row_idx].to(self.device)
            return utils.get_sparse_idx(self.Ab,row_idx).to_dense().to(self.device)
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_activation_pattern(self,x):
        '''
        Get VQ code
        '''
        x = self.flat2img(x,self.input_shape) #reshape before forward
        pre_act = self.layer.forward(x)
        
        if self.has_bn:
            pre_act = self.bn.forward(pre_act)
        
        pre_act = self.img2flat(pre_act)
        
        if not self.has_act:
            q = torch.ones_like(pre_act)
        elif self.act_name == 'lrelu':
            q = (pre_act>0)*1. + (pre_act<=0)*self.act_nslope
        elif self.act_name == 'relu':
            q = (pre_act>0)*1.
        
        return q
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_intersection_pattern(self,x):
        '''
        Get sign of x wrt layer hyperplanes
        '''
        x = self.flat2img(x,self.input_shape)
        pre_act = self.layer.forward(x)
        
        if self.has_bn:
            pre_act = self.bn.forward(pre_act)
            
        pre_act = self.img2flat(pre_act)
            
        return (pre_act>0)*1.
    
    @torch.no_grad()
    def img2flat(self,x):
        return x.reshape(x.shape[0],-1)
    
    @torch.no_grad()
    def flat2img(self,x,shape):
        return x.reshape(x.shape[0],shape[0],shape[1],shape[2])
    
#     @torch.no_grad()
#     @torch.jit.script_method TODO: Make these methods jittable
    @torch.no_grad()
    def forward(self,x):
        '''
        Follows order Linear->BN->activation by default
        '''
        x = self.flat2img(x,self.input_shape)

        x = self.layer.forward(x)
        
        if self.has_bn:
            x = self.bn.forward(x)
        if self.has_act:
            x = self.act.forward(x)
        
        return self.img2flat(x)
    
# class conv2d(torch.nn.Module):
#     """
#     Wrapper for a torch.nn.Linear layer with batchnorm and activation function
#     Module forward is in sequence Linear->BN->Activation

#     Attributes
#     ----------
    
#     has_act,has_bn : bool:
#         whether layer contains batchnorm or activation function
#     act,bn : torch.nn.Module
#         Batchnorm and activation torch module
#     name : str
#         name of module
#     device : {'cuda','cpu'}
#         device to store parameters in    
    
#     Methods
#     -------
#     TODO: complete docstring
    
#     """
    
#     def __init__(self,conv2d_layer,input_shape,
#                  act_layer=None,bn_layer=None,name='conv2d_layer',
#                  device='cuda', dtype=torch.float32):
#         '''
#         Wrapper for linear layer with batchnorm and/or ReLU activation
#         Follows order Linear->BN->Activation
#         '''
#         super(conv2d, self).__init__()
        
#         self.device = device
#         self.name = name
#         self.dtype = dtype
        
#         self.input_shape = torch.tensor(input_shape)
#         self.layer = conv2d_layer.to(self.device).type(self.dtype)
        
#         #enforcing symmetric
#         assert self.layer.stride[0] == self.layer.stride[1] 
#         assert self.layer.padding[0] == self.layer.padding[1]
#         assert self.layer.kernel_size[0] == self.layer.kernel_size[1]
#         assert self.layer.kernel_size[0] - 2*self.layer.padding[0] == 1 ## padding same
        
#         self.n_kernels = self.layer.weight.shape[0]
#         self.stride = self.layer.stride[0]
        
#         self.output_shape = torch.tensor(
#             (self.n_kernels,
#              self.input_shape[1]/self.stride,
#              self.input_shape[2]/self.stride)
#         ).type(torch.int64)
        
#         # check if layer has act and bn
#         self.has_act = act_layer is not None
#         self.has_bn = bn_layer is not None       
        
#         if self.has_act:
#             self.add_act(act_layer)
#         else:
#             self.act = lambda x:x
        
#         if self.has_bn:
#             self.add_bn(bn_layer)
#         else:
#             self.bn = lambda x:x
            
# #         self.Ab = self.prepare_weights()
            
        
#     @torch.no_grad() #TODO: Make these methods jit ignore
#     def add_bn(self,bn_layer):
        
#         self.has_bn = True
        
#         self.bn = bn_layer.to(self.device).type(self.dtype)
#         self.bn_rmean = self.bn.__dict__['_buffers']['running_mean']
#         self.bn_rvar = self.bn.__dict__['_buffers']['running_var']
#         self.bn_gamma = self.bn.__dict__['_parameters']['weight']
#         self.bn_beta = self.bn.__dict__['_parameters']['bias']
#         self.bn_eps = self.bn.__dict__['eps']
        
#         self.Ab = self.prepare_weights()
        
    
#     @torch.no_grad() #TODO: Make these methods jit ignore
#     def add_act(self,act_layer):    
        
#         self.has_act = True
        
#         self.act = act_layer.to(self.device).type(self.dtype)
        
#         if type(self.act) == torch.nn.modules.activation.LeakyReLU:
#             self.act_name = 'lrelu'
#             self.act_nslope = self.act.__dict__['negative_slope']
            
#         elif type(self.act) == torch.nn.modules.activation.ReLU:
#             self.act_name = 'relu'
            
#         else:
#             raise NotImplementedError('Activation func not supported')
            
#     @torch.no_grad() #TODO: make jittable
#     def prepare_weights(self):
#         '''
#         Get matrix representation of conv
#         '''
    
#         identity = torch.eye(
#             torch.prod(
#                 self.input_shape
#             ).type(torch.int64), device='cpu', dtype=torch.float32
#         ).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2]
#                  )
        
#         self.layer.type(torch.float32)
        
#         if self.has_bn:
            
#             c = self.bn_gamma/(self.bn_rvar + self.bn_eps)**.5
#             b = self.bn_beta - self.bn_rmean*c
            
#             c = c.type(torch.float32)
#             b = b.type(torch.float32)
            
#             method = lambda x: c[None,:,None,None]*self.layer.forward(x)
        
#         else:
#             method = self.layer.forward
        
        
#         output = _batched_gpu_op(
#             method = method, data=identity, batch_size=2048*2, dtype=torch.float32,
#             out_size=(identity.shape[0],self.output_shape[0],self.output_shape[1],self.output_shape[2]) 
#         )
        
# #         output = self.layer.forward(identity)
    
#         W = output.reshape(-1, torch.prod(self.output_shape).type(torch.int64)).T

#         if self.layer.bias is not None:
            
#             bias = self.layer.bias.data
            
#             if self.has_bn:
#                 bias = c*self.layer.bias.data + b
                
#             b = torch.stack([torch.ones(self.output_shape[1],self.output_shape[2]
#                                        ).type(torch.float32).to(self.device) * bi for bi in bias])
#             b = b.reshape(torch.prod(self.output_shape).type(torch.int64),1)

#         else:

#             bias = torch.zeros(self.output_shape[0]).type(torch.float32).to(self.device)
            
#             if self.has_bn:
#                 bias += b
                
#             b = torch.stack([torch.ones(self.output_shape[1],self.output_shape[2]
#                                        ).type(torch.float32).to(self.device) * bi for bi in bias])
#             b = b.reshape(torch.prod(self.output_shape).type(torch.int64),1)
        
#         b = b.to(W.device)
#         self.layer.type(self.dtype)
        
#         try: ## sparse matrix not allowed for very large matrices
#             W = torch.hstack([W,b]).to_sparse().cpu().type(self.dtype) 
#         except RuntimeError:
#             warnings.warn('Abw too big for creating sparse matrix')
#             W = torch.hstack([W,b]).cpu().type(self.dtype)
#         return W
    
#     @torch.no_grad() #TODO: Make these methods jittable
#     def get_weights(self,row_idx=None,dtype=torch.float64):
#         '''
#         Returns Ab = [Aw|bw]
#         if row_idx specified, only returns those hyps/rows
#         defualt stores Ab in cpu and returns gpu during calls
#         '''
        
#         if not hasattr(self,'Ab'):
#             ## get matrix representation of weights
#             self.Ab = self.prepare_weights()
                
# #         if row_idx is None:
# #             return self.Ab.to_dense().to(self.device) 
# #         else:
# #             return self.Ab.to_dense()[row_idx].to(self.device)
        
#         if row_idx is None:
#             return self.Ab.type(dtype).to(self.device)
#         else:
# #             return self.Ab[row_idx].to(self.device)
#             return utils.get_sparse_idx(self.Ab,row_idx).to_dense().to(self.device)
    
#     @torch.no_grad() #TODO: Make these methods jittable
#     def get_activation_pattern(self,x):
#         '''
#         Get VQ code
#         '''
#         x = self.flat2img(x,self.input_shape) #reshape before forward
#         pre_act = self.layer.forward(x)
        
#         if self.has_bn:
#             pre_act = self.bn.forward(pre_act)
        
#         pre_act = self.img2flat(pre_act)
        
#         if not self.has_act:
#             q = torch.ones_like(pre_act)
#         elif self.act_name == 'lrelu':
#             q = (pre_act>0)*1. + (pre_act<=0)*self.act_nslope
#         elif self.act_name == 'relu':
#             q = (pre_act>0)*1.
        
#         return q
    
#     @torch.no_grad() #TODO: Make these methods jittable
#     def get_intersection_pattern(self,x):
#         '''
#         Get sign of x wrt layer hyperplanes
#         '''
#         x = self.flat2img(x,self.input_shape)
#         pre_act = self.layer.forward(x)
        
#         if self.has_bn:
#             pre_act = self.bn.forward(pre_act)
            
#         pre_act = self.img2flat(pre_act)
            
#         return (pre_act>0)*1.
    
#     @torch.no_grad()
#     def img2flat(self,x):
#         return x.reshape(x.shape[0],-1)
    
#     @torch.no_grad()
#     def flat2img(self,x,shape):
#         return x.reshape(x.shape[0],shape[0],shape[1],shape[2])
    
# #     @torch.no_grad()
# #     @torch.jit.script_method TODO: Make these methods jittable
#     @torch.no_grad()
#     def forward(self,x):
#         '''
#         Follows order Linear->BN->activation by default
#         '''
#         x = self.flat2img(x,self.input_shape)

#         x = self.layer.forward(x)
        
#         if self.has_bn:
#             x = self.bn.forward(x)
#         if self.has_act:
#             x = self.act.forward(x)
        
#         return self.img2flat(x)
    
    
class avgpool2d(torch.nn.Module):
    """
    Wrapper for a torch.nn.AvgPool2d layer 
    Module forward is in sequence Linear->BN->Activation

    Attributes
    ----------
    
    has_act,has_bn : bool:
        whether layer contains batchnorm or activation function
    act,bn : torch.nn.Module
        Batchnorm and activation torch module
    name : str
        name of module
    device : {'cuda','cpu'}
        device to store parameters in    
    
    Methods
    -------
    TODO: complete docstring
    
    """
    
    def __init__(self,pool_layer,input_shape,name='avgpool2d_layer',
                 device='cuda',dtype=torch.float32):
        '''
        Wrapper for linear layer with batchnorm and/or ReLU activation
        Follows order Linear->BN->Activation
        '''
        super(avgpool2d, self).__init__()
        
        self.device = device
        self.name = name
        self.dtype = dtype
        
        self.input_shape = torch.tensor(input_shape)
        self.layer = pool_layer.to(self.device).type(self.dtype)
        
        #enforcing symmetric
        assert type(self.layer.stride) == int
        
        # if self.layer.kernel_size == 2:
        #     assert self.layer.padding == 0
        # elif self.layer.kernel_size>2:
        #     assert self.layer.kernel_size[0] - 2*self.layer.padding[0] == 1

        
        # assert self.layer.stride[0] == self.layer.stride[1] 
        # assert self.layer.padding[0] == self.layer.padding[1]
        # assert self.layer.kernel_size[0] == self.layer.kernel_size[1]
        # assert self.layer.kernel_size[0] - 2*self.layer.padding[0] == 1 ## padding same                         
                     
        
        self.output_shape = torch.tensor(
            (self.input_shape[0],
             self.input_shape[1]/self.layer.stride,
             self.input_shape[2]/self.layer.stride)
        ).type(torch.int64)
        
        
        self.Ab = self.prepare_weights()
    
    @torch.no_grad() #TODO: Make these methods jit ignore
    def prepare_weights(self):
        
        identity = torch.eye(
            torch.prod(
                self.input_shape
            ).type(torch.int64), device='cpu', dtype=torch.float32
        ).reshape(-1,self.input_shape[0],self.input_shape[1],self.input_shape[2]
                 )
        
        self.layer.type(torch.float32)
        output = _batched_gpu_op(
            method = self.layer.forward, data=identity, batch_size=2048*2,workers=0,
            dtype=torch.float32,
            out_size=(identity.shape[0],self.output_shape[0],self.output_shape[1],self.output_shape[2]) 
        )
        
        self.layer.type(self.dtype)

        W = output.reshape(-1, torch.prod(self.output_shape).type(torch.int64)).T
        b = torch.zeros(W.shape[0],1,device='cpu',dtype=torch.float32)
        
        return torch.hstack([W,b]).to_sparse().cpu().type(self.dtype)
            
    @torch.no_grad() #TODO: Make these methods jittable
    def get_weights(self,row_idx=None,dtype=torch.float64):
        '''
        Returns Ab = [Aw|bw]
        if row_idx specified, only returns those hyps/rows
        defualt stores Ab in cpu and returns gpu during calls
        '''
        
        if not hasattr(self,'Ab'):
            ## get matrix representation of weights
            self.Ab = self.prepare_weights()
                
#         if row_idx is None:
#             return self.Ab.to_dense().to(self.device) 
#         else:
#             return self.Ab.to_dense()[row_idx].to(self.device)
        
        if row_idx is None:
            return self.Ab.type(dtype).to(self.device) 
        else:
#             return self.Ab[row_idx].to(self.device)
            return utils.get_sparse_idx(self.Ab,row_idx).to_dense().to(self.device)
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_activation_pattern(self,x):
        '''
        Get VQ code
        '''
        return torch.ones(x.shape[0],
                          torch.prod(self.output_shape), device=self.device).type(self.dtype)
    
    @torch.no_grad() #TODO: Make these methods jittable
    def get_intersection_pattern(self,x):
        '''
        Get sign of x wrt layer hyperplanes
        '''
        return self.get_activation_pattern(x)
    
    @torch.no_grad()
    def img2flat(self,x):
        return x.reshape(x.shape[0],-1)
    
    @torch.no_grad()
    def flat2img(self,x,shape):
        return x.reshape(x.shape[0],shape[0],shape[1],shape[2])
    
#     @torch.no_grad()
#     @torch.jit.script_method TODO: Make these methods jittable
    @torch.no_grad()
    def forward(self,x):
        '''
        Follows order Linear->BN->activation by default
        '''
        x = self.flat2img(x,self.input_shape)
        x = self.layer.forward(x)
        return self.img2flat(x)
