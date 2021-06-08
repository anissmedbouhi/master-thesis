import torch
from scipy import stats


#Mean Square Error - Loss for an Auto-Encoder: Reconstruction loss
def MSE(recon_X, X):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    N = X.shape[0]
    MSE = torch.sum(torch.pow(recon_X - X, 2))
    return MSE/N


#Loss for a Variational Auto-Encoder with a Gaussian Prior:
#Reconstruction loss (Mean Square Error) and KL-divergence
def MSEKLD(recon_X, X, mu, log_var, beta):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    N = X.shape[0]
    MSE = torch.sum(torch.pow(recon_X - X, 2))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE/N + beta*KLD/N


# Compute the LogProb of a non-Gaussian prior given a x and the parameters of the prior
def LogProbPrior(x, weights, means, covariances):
    s = 0
    for i, weight in enumerate(weights):
        mean = means[i]
        covariance = covariances[i]
        s += weight * torch.exp(
            # if full then covariance_matrix = covariance
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc = mean,
                covariance_matrix = torch.diag(covariance)
                ).log_prob(x)
        )
    return torch.log(s)
    

#Loss for a Variational Auto-Encoder with a non-Gaussian Prior:
#(for example with a Gaussian Mixture Model as a Prior,
#the KL-divergence cannot be computed easily and a Monte Carlo simulation is needed)
#Reconstruction loss (Mean Square Error) and KL-divergence estimated using
#a Monte Carlo simulation sampling N times inspired by the reparametrization trick 
def MSESAMPLINGKLD(recon_X, X, mu, log_var, LogProbEncoder, LogProbPrior, weights, means, covariances): # 1 sample in MC
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    N = X.shape[0]
    MSE = torch.sum(torch.pow(recon_X - X, 2))

    # sampling to estimate KLD
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    samples = eps.mul(std).add_(mu)

    A = torch.FloatTensor([ f.log_prob(samples[i]) for i, f in enumerate(LogProbEncoder) ]).cuda()
    B = LogProbPrior(samples, weights, means, covariances)
    KLD = torch.sum( A - B )

    return MSE/N + KLD/N


def MSESAMPLINGKLDNEW(recon_X, X, mu, log_var, samples_kld, LogProbEncoder, LogProbPrior): # samples_kld is the number of samples in MC, used in VAE with GMM prior
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    N = X.shape[0]
    MSE = torch.sum(torch.pow(recon_X - X, 2))

    assert samples_kld > mu.shape[0]
    number = samples_kld//mu.shape[0]

    def get_samples():
        # sampling to estimate KLD
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        samples = eps.mul(std).add_(mu)

        A = torch.FloatTensor([ f.log_prob(samples[i]) for i, f in enumerate(LogProbEncoder) ]).cuda()
        B = LogProbPrior(samples)

        return A, B
    
    KLD = 0
    for _ in range(number):
        A, B = get_samples()
        KLD += torch.sum( A - B )

    return MSE/N + KLD/(N*number)
   

#This is the simplicial regularization
# f can be the encoder or the decoder
# selection1 should be the data points in the input space of f for a specific simplicial complex
# selection2 should be the data points in the output space of f for the same simplicial complex, i.e. the images of selection1 by f
def SimplicialLossCode(f, selection1, selection2, lambdas):

    assert len(selection1) == len(selection2) == len(lambdas)

    n = len(lambdas)
    s = 0
    for i in range(n):
        s += MSE( f( lambdas[i] @ selection1[i], sample = True ), lambdas[i] @ selection2[i] )

    return s / n


#As previously but used if there is a probability assigned to each simplex, for example the probabilities from the Fuzzy simplicial set
def SimplicialLossCodeProbas(f, selection1, selection2, lambdas, probas):

    assert len(selection1) == len(selection2) == len(lambdas) == len(probas)

    n = len(lambdas)
    s = 0
    for i in range(n):
        s += probas[i] * MSE( f( lambdas[i] @ selection1[i], sample = True ), lambdas[i] @ selection2[i] )
    
    return s / n
