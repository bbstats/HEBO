from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from bo.kernels import *
from botorch.fit import fit_gpytorch_model

# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, kern, likelihood,
                 outputscale_constraint,
                 ard_dims, cat_dims=None):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.dim = train_x.shape[1]
        self.ard_dims = ard_dims
        self.cat_dims = cat_dims
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(kern, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)  # , cat_dims, int_dims)
        return MultivariateNormal(mean_x, covar_x)


def train_gp(train_x, train_y, use_ard, num_steps, kern='transformed_overlap', hypers={},
             noise_variance=None,
             cat_configs=None,
             antigen=None,
             search_strategy='local',
             **params):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized.
    （train_x, train_y）: pairs of x and y (trained)
    noise_variance: if provided, this value will be used as the noise variance for the GP model. Otherwise, the noise
        variance will be inferred from the model.
    """
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    # Create hyper parameter bounds
    if noise_variance is None:
        noise_variance = 0.005
        noise_constraint = Interval(1e-6, 0.1)
    else:
        if np.abs(noise_variance) < 1e-6:
            noise_variance = 0.05
            noise_constraint = Interval(1e-6, 0.1)
        else:
            noise_constraint = Interval(0.99 * noise_variance, 1.01 * noise_variance)
    if use_ard:
        lengthscale_constraint = Interval(0.01, 0.5)
    else:
        lengthscale_constraint = Interval(0.01, 2.5)  # [0.005, sqrt(dim)]

    outputscale_constraint = Interval(0.5, 5.)

    # Create models
    if search_strategy=='glocal':
        # Remove constraints for better GP fit
        likelihood = GaussianLikelihood().to(device=train_x.device, dtype=train_y.dtype)
    else:
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device,
                                                                              dtype=train_y.dtype)

    ard_dims = train_x.shape[1] if use_ard else None

    if kern == 'overlap':
        kernel = CategoricalOverlap(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'transformed_overlap':
        if search_strategy == 'glocal':
            kernel = TransformedCategorical(ard_num_dims=ard_dims)
        else:
            kernel = TransformedCategorical(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, )
    elif kern == 'ordinal':
        kernel = OrdinalKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, config=cat_configs)
    elif kern == 'ssk':
        kernel = FastStringKernel(seq_length=train_x.shape[1], alphabet_size=params['alphabet_size'], device=train_x.device)
    elif kern in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
        from bo.utils import BERTFeatures, batch_iterator
        from einops import rearrange
        bert = BERTFeatures(params['BERT_model'], params['BERT_tokeniser'])
        nm_samples = train_x.shape[0]
        if nm_samples > params['BERT_batchsize']:
            reprsn1 = []
            for x in batch_iterator(train_x, params['BERT_batchsize']):
                features1 = bert.compute_features(x)
                reprsn1.append(features1)
            reprsn1 = torch.cat(reprsn1, 0)
        else:
            reprsn1 = bert.compute_features(train_x)
        reprsn1 = rearrange(reprsn1, 'b l d -> b (l d)')
        if kern in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
            from joblib import dump, load
            pca = load(f"/nfs/aiml/asif/CDRdata/pca/{antigen}_pca.joblib")
            scaler = load(f"/nfs/aiml/asif/CDRdata/pca/{antigen}_scaler.joblib")
            reprsn1 = torch.from_numpy(pca.transform(scaler.transform(reprsn1.cpu().numpy())))
        train_x = reprsn1.clone()
        del reprsn1, bert
        ard_dims = train_x.shape[1] if use_ard else None
        if kern in ['rbfBERT', 'rbf-pca-BERT']:
            kernel = BERTWarpRBF(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
        else:
            kernel = BERTWarpCosine(lengthscale_constraint=lengthscale_constraint, ard_num_dims=None)
    elif kern == 'rbf':
        kernel = RBFKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims)
    else:
        raise ValueError('Unknown kernel choice %s' % kern)

    if search_strategy == 'glocal':
        # Similarly remove constraints
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kern=kernel,
            outputscale_constraint=None,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)
    else:
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            kern=kernel,
            outputscale_constraint=outputscale_constraint,
            ard_dims=ard_dims,
        ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if search_strategy == 'glocal':
        if hypers:
            model.load_state_dict(hypers)
        fit_gpytorch_model(mll)
    else:
        # Initialize model hypers
        if hypers:
            model.load_state_dict(hypers)
        else:
            hypers = {}
            hypers["covar_module.outputscale"] = 1.0
            if not (isinstance(kernel, FastStringKernel) or isinstance(kernel, CosineKernel)):
                hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
            hypers["likelihood.noise"] = noise_variance if noise_variance is not None else 0.005
            model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.03)

        for i in range(num_steps):
            optimizer.zero_grad()
            output = model(train_x, )
            loss = -mll(output, train_y).float()
            loss.backward()
            #print(f"Loss Step {i} = {loss.item()}")
            optimizer.step()

    # Switch to eval mode
    model.eval()
    return model