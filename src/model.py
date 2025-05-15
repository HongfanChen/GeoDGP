
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel, SpectralDeltaKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
import torch
class MaternLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=256, mean_type='constant', nu=0.5):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(MaternLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(MaternKernel(nu))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class SpecetralDeltaLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(SpecetralDeltaLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(SpectralDeltaKernel(
            num_dims=input_dims,
            num_deltas=1000
        ))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
class DeepGaussianProcess(DeepGP):
    def __init__(self,
                 input_shape,
                 DEVICE,
                 num_hidden1_dims = 20,
                 num_hidden2_dims = 10,
                 num_hidden3_dims = 10):
        hidden_layer_1 = MaternLayer(
            input_dims=input_shape,
            output_dims=num_hidden1_dims,
            num_inducing=256,
            mean_type='linear',
            nu=0.5
        )
        hidden_layer_2 = MaternLayer(
            input_dims=hidden_layer_1.output_dims,
            output_dims=num_hidden2_dims,
            num_inducing=128,
            mean_type='linear',
            nu=1.5
        )
        hidden_layer_3 = MaternLayer(
            input_dims=hidden_layer_2.output_dims,
            output_dims=num_hidden3_dims,
            num_inducing=128,
            mean_type='linear',
            nu=1.5
        )
        last_layer = SpecetralDeltaLayer(
            input_dims=hidden_layer_3.output_dims,
            output_dims=None,
            num_inducing=128,
            mean_type='linear',
        )
        super().__init__()

        self.hidden_layer1 = hidden_layer_1
        self.hidden_layer2 = hidden_layer_2
        self.hidden_layer3 = hidden_layer_3
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.device = DEVICE
    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer1(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        hidden_rep3 = self.hidden_layer3(hidden_rep2)
        output = self.last_layer(hidden_rep3)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for batch in test_loader:
                x_batch = batch['X'].to(self.device)
                y_batch = batch['Y'].to(self.device)
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)