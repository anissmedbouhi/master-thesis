from callers import *

### Legend of the parameters:
# N: number of simplices chosen randomly at each iteration, choose a multiple of 8 to optimize GPU calculations
# weight_encoder_simplicial_reg: weigth of the simplicial regularization of the encoder
# weight_decoder_simplicial_reg: weigth of the simplicial regularization of the decoder
# beta: weight of the KL-divergence like in a Beta-VAE
# max_epoch is the total number of epochs
# number_samples_lambdas: number of samples used to do the expectation of the symmetric Dirichlet distribution, the expectation is a Monte-Carlo simulation
# alpha: parameter alpha of a symmetric Dirichlet distribution over the simplex. If alpha = 1, it is equivalent to a uniform distribution over simplex, and as alpha tends towards 0, the distribution becomes more concentrated on the vertices of the simplex.

# (Beta) Variational Auto-Encoder with the possibility to use a Gaussian Mixture Model prior instead of a Gaussin prior, and with the usual loss
def vae(DataLoaderTrain, DataLoaderValidation, model, optimizer, LOSSES, weights_prior=None, means_prior=None, covariances_prior=None, GMM_prior=False, VALIDATION = True, beta = 1, max_epoch = 100):

  C=max_epoch*len(DataLoaderTrain) # max_epoch* nmbr batch
  counter=0
  for epoch in range(max_epoch):
    
    print(' epoch:', epoch)

    for batch, (x, y, id_batch) in tqdm(enumerate(DataLoaderTrain)):

      #beta=1+9*counter/C ## low beta->good reconstruction vs high beta->good disentanglement
      counter+=1
      model = model.train()

      recon_x, Z, mu, logvar = model(x)

      if GMM_prior == False:
        loss = MSEKLD(recon_x, x, mu, logvar, beta)
      else:
        LogProbEncoder = [ torch.distributions.multivariate_normal.MultivariateNormal(
            loc = mu[i],
            covariance_matrix = torch.diag(torch.exp(logvar[i]))
            ) for i in range(len(mu)) ]
        loss = MSESAMPLINGKLD(recon_x, x, mu, logvar, LogProbEncoder, LogProbPrior, weights_prior, means_prior, covariances_prior)        

      ## topoloss = TL.compute(x = x, latent = Z, latent_norm = model.latent_norm)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # update the train loss
      LOSSES['train'].append(loss.item())

      if VALIDATION == True: # To avoid too many computations you can switch off the validation in the arguments when calling the function

        model = model.eval()
        S = []
        C = 0
        for batch, (Xval, yval) in enumerate(DataLoaderValidation):
            
          recon_Xval, Zval, mu, logvar = model(Xval)
    
          if GMM_prior == False:
            loss_ = MSEKLD(recon_Xval, Xval, mu, logvar, beta)
          else:
            LogProbEncoder = [ torch.distributions.multivariate_normal.MultivariateNormal(
                loc = mu[i],
                covariance_matrix = torch.diag(torch.exp(logvar[i]))
                ) for i in range(len(mu)) ]
            loss_ = MSESAMPLINGKLD(recon_Xval, Xval, mu, logvar, LogProbEncoder, LogProbPrior, weights_prior, means_prior, covariances_prior)            

          S.append( loss_.item() * Xval.shape[0] )
          C += Xval.shape[0]

        # update the validation loss
        LOSSES['validation'].append(np.sum(S)/C)


def InvMap_VAE(DataLoaderTrain, DataLoaderValidation, model, optimizer, LOSSES, VALIDATION = True, beta = 1, weight_loss_embedding = 1.0, max_epoch = 100):

  C=max_epoch*len(DataLoaderTrain) # max_epoch* nmbr batch
  counter=0
  for epoch in range(max_epoch):

      print(' epoch:', epoch)
      
      for batch, (X_batch, y_batch, Z_embedding_batch) in tqdm(enumerate(DataLoaderTrain)):

          #beta=1+9*counter/C ## low beta->good reconstruction vs high beta->good disentanglement
          counter+=1
          model = model.train()

          recon_X_batch, Z_batch, mu, logvar = model(X_batch)
          loss1 = MSEKLD(recon_X_batch, X_batch, mu, logvar, beta)

          loss2 = MSE(Z_batch, Z_embedding_batch)

          loss = loss1 + weight_loss_embedding*loss2


          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update the train loss
          LOSSES['train'].append(loss.item())

          if VALIDATION == True:
            model = model.eval()
            S = []
            C = 0
            for batch, (Xval, yval) in enumerate(DataLoaderValidation):
                
                recon_Xval, Zval, mu, logvar = model(Xval)
                #loss_ = MSE(recon_Xval, Xval)

                loss_ = MSEKLD(recon_Xval, Xval, mu, logvar, beta)

                S.append( loss_.item() * Xval.shape[0] )
                C += Xval.shape[0]

            # update the validation loss
            LOSSES['validation'].append(np.sum(S)/C)


# Adaptation of the "Simplicial AutoEncoder" to a (Beta) Variational Auto-Encoder and giving the possibility to use a Gaussian Mixture Model Prior
# using the Fuzzy simplicial complex built from the embedding via UMAP for example
# To learn more about the "Simplicial AutoEncoder" of Jose Daniel Gallego Posada: https://esc.fnwi.uva.nl/thesis/centraal/files/f1267161058.pdf
def Simplicial_VAE(X, DataLoaderTrain, model, optimizer, Z_embedding, dictKZ, LOSSES, weights_prior=None, means_prior=None, covariances_prior=None, GMM_prior=False, N = 64, weight_encoder_simplicial_reg = 10.0, weight_decoder_simplicial_reg = 10.0, beta = 1, max_epoch = 100, number_samples_lambdas = 10, alpha = 1.0):
  
  dictKZkeys = list(dictKZ.keys())

  for epoch in range(max_epoch):

      print(' epoch:', epoch)

      for batch, (x, y, id_batch) in tqdm(enumerate(DataLoaderTrain)):

          model = model.train()

          recon_x, z, mu, logvar = model(x)

          if GMM_prior == False:
            loss1 = MSEKLD(recon_x, x, mu, logvar, beta)
          else:
            LogProbEncoder = [ torch.distributions.multivariate_normal.MultivariateNormal(
                loc = mu[i],
                covariance_matrix = torch.diag(torch.exp(logvar[i]))
                ) for i in range(len(mu)) ]
            loss1 = MSESAMPLINGKLD(recon_x, x, mu, logvar, LogProbEncoder, LogProbPrior, weights_prior, means_prior, covariances_prior)            

          if N == None: # if considering all the simplices
            choicesZ = dictKZkeys # too long computationnaly...
          else: # if considering only N simplices chosen randomly
            choicesZ = np.random.choice(dictKZkeys, N, replace = False) 

          probas = [] # list of the probabilities of each simplex
          for i in choicesZ:
            probas.append(dictKZ[i])

          choicesZ = np.repeat(choicesZ, number_samples_lambdas) #instead of doing a for loop for the lambdas sampling
          probas = np.repeat(probas, number_samples_lambdas)
          simplicesX = [ cuda(X[name, ...]) for name in choicesZ ]       
          simplicesZ = [ cuda(Z_embedding[name, ...]) for name in choicesZ ]

          lambdas = [ cuda(np.random.dirichlet(len(name) * [alpha])) for name in choicesZ ]

          loss2 = SimplicialLossCodeProbas(f = model.encoder, selection1 = simplicesX, selection2 = simplicesZ, lambdas = lambdas, probas = probas)
          loss3 = SimplicialLossCodeProbas(f = model.decoder, selection1 = simplicesZ, selection2 = simplicesX, lambdas = lambdas, probas = probas)
          
          loss = loss1 + weight_encoder_simplicial_reg * loss2 + weight_decoder_simplicial_reg * loss3

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update the train loss
          LOSSES['train'].append(loss.item())

# Modification in order to not use any embedding: simplicial regularization using a Fuzzy simplicial complex built directly from the input space data points
def Fuzzy_Simplicial_VAE(X, DataLoaderTrain, model, optimizer, dictKX, LOSSES, N = None, weight_encoder_simplicial_reg = 10.0, weight_decoder_simplicial_reg = 10.0, beta = 1, max_epoch = 100, number_samples_lambdas = 10, alpha = 1.0):
  
  dictKXkeys = list(dictKX.keys())

  for epoch in range(max_epoch):

      print(' epoch:', epoch)

      for batch, (x, y, id_batch) in tqdm(enumerate(DataLoaderTrain)):

          model = model.train()

          recon_x, z, mu, logvar = model(x)

          loss1 = MSEKLD(recon_x, x, mu, logvar, beta)

          if N == None: # if considering all the simplices
            choicesX = dictKXkeys # too long computationnaly...
          else: # if considering only N simplices chosen randomly
            choicesX = np.random.choice(dictKXkeys, N, replace = False) 

          probas = [] # list of the probabilities of each simplex
          for i in choicesX:
            probas.append(dictKX[i])

          choicesX = np.repeat(choicesX, number_samples_lambdas) #instead of doing a for loop for the lambdas sampling
          probas = np.repeat(probas, number_samples_lambdas)
          simplicesX = [ cuda(X[name, ...]) for name in choicesX ]
          
          #with mu of encoder(x): simplicesZ = [ model.encoder(cuda(X[name, ...]))[0] for name in choicesX ]
          simplicesZ = [ model.encoder(cuda(X[name, ...]), sample = True) for name in choicesX ] # with sampling z=encoder(x)

          lambdas = [ cuda(np.random.dirichlet(len(name) * [alpha])) for name in choicesX ]

          loss2 = SimplicialLossCodeProbas(f = model.encoder, selection1 = simplicesX, selection2 = simplicesZ, lambdas = lambdas, probas = probas)
          loss3 = SimplicialLossCodeProbas(f = model.decoder, selection1 = simplicesZ, selection2 = simplicesX, lambdas = lambdas, probas = probas)
          
          loss = loss1 + weight_encoder_simplicial_reg * loss2 + weight_decoder_simplicial_reg * loss3

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update the train loss
          LOSSES['train'].append(loss.item())

def Witness_Simplicial_VAE(X, DataLoaderTrain, model, optimizer, K_witnesscomplex, LOSSES, N = None, weight_encoder_simplicial_reg = 10.0, weight_decoder_simplicial_reg = 10.0, beta = 1, max_epoch = 100, number_samples_lambdas = 10, alpha = 1.0):
  
  for epoch in range(max_epoch):

      print(' epoch:', epoch)

      for batch, (x, y, id_batch) in tqdm(enumerate(DataLoaderTrain)):

          model = model.train()

          recon_x, z, mu, logvar = model(x)

          loss1 = MSEKLD(recon_x, x, mu, logvar, beta)

          if N == None: # if considering all the simplices of K_witnesscomplex
            choicesX = K_witnesscomplex
          else: # if considering only N simplices chosen randomly from K_witnesscomplex
            choicesX = np.random.choice(K_witnesscomplex, N, replace = False) 

          choicesX = np.repeat(choicesX, number_samples_lambdas) #instead of doing a for loop for the lambdas sampling
          simplicesX = [ cuda(X[name, ...]) for name in choicesX ]
          
          #with mu of encoder(x): simplicesZ = [ model.encoder(cuda(X[name, ...]))[0] for name in choicesX ]
          simplicesZ = [ model.encoder(cuda(X[name, ...]), sample = True) for name in choicesX ] # with sampling z=encoder(x)

          lambdas = [ cuda(np.random.dirichlet(len(name) * [alpha])) for name in choicesX ]

          loss2 = SimplicialLossCode(f = model.encoder, selection1 = simplicesX, selection2 = simplicesZ, lambdas = lambdas)
          loss3 = SimplicialLossCode(f = model.decoder, selection1 = simplicesZ, selection2 = simplicesX, lambdas = lambdas)

          loss = loss1 + weight_encoder_simplicial_reg * loss2 + weight_decoder_simplicial_reg * loss3

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update the train loss
          LOSSES['train'].append(loss.item())


def Witness_Complexes_Simplicial_VAE(X, DataLoaderTrain, model, optimizer, K_witnesscomplexes, LOSSES, N = None, weight_encoder_simplicial_reg = 10.0, weight_decoder_simplicial_reg = 10.0, beta = 1, max_epoch = 100, number_samples_lambdas = 10, alpha = 1.0):
  
  for epoch in range(max_epoch):

      print(' epoch:', epoch)

      for batch, (x, y, id_batch) in tqdm(enumerate(DataLoaderTrain)):

          model = model.train()

          recon_x, z, mu, logvar = model(x)

          loss1 = MSEKLD(recon_x, x, mu, logvar, beta)

          # K_witnesscomplexes[batch] is the Witness Complexe built such that the landmarks are the data of the current batch and the witnesses are the whole X

          if N == None: # if considering all the simplices for each Witness Complex of K_witnesscomplexes
            choicesX = K_witnesscomplexes[batch]
          else: # if considering N simplices chosen randomly from K_witnesscomplexes[batch]
            choicesX = np.random.choice(K_witnesscomplexes[batch], N, replace = False) 

          choicesX = np.repeat(choicesX, number_samples_lambdas) #instead of doing a for loop for the lambdas sampling
          simplicesX = [ cuda(X[name, ...]) for name in choicesX ]
          
          #with mu of encoder(x): simplicesZ = [ model.encoder(cuda(X[name, ...]))[0] for name in choicesX ]
          simplicesZ = [ model.encoder(cuda(X[name, ...]), sample = True) for name in choicesX ] # with sampling z=encoder(x)

          lambdas = [ cuda(np.random.dirichlet(len(name) * [alpha])) for name in choicesX ]

          loss2 = SimplicialLossCode(f = model.encoder, selection1 = simplicesX, selection2 = simplicesZ, lambdas = lambdas)
          loss3 = SimplicialLossCode(f = model.decoder, selection1 = simplicesZ, selection2 = simplicesX, lambdas = lambdas)

          loss = loss1 + weight_encoder_simplicial_reg * loss2 + weight_decoder_simplicial_reg * loss3

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update the train loss
          LOSSES['train'].append(loss.item())