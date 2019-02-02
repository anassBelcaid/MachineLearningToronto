from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  # mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  mu = KMeans(x, K, 5)
  
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in range(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in range(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print('Iter %d logProb %.5f' % (i, logProbX[i]))

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in range(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in range(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def q3():
  iters = 10
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  print("without initialization")
  K = 20
  p, mu, vary, log = mogEM(inputs_train, K, iters,minVary)
  # plt.savefig("mog_with_initialization.png")


  #Now with k means 

  

  input('Press Enter to continue.')
def q4():
  iters = 10
  minVary = 0.01
  numComponents = np.array([2,3, 5,10, 13,15,20,23, 25])
  L = len(numComponents)
  errorTrain = np.zeros(L)
  errorTest = np.zeros(L)
  errorValidation = np.zeros(L)
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in range(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, log2 = mogEM(train2, K, iters,minVary)

    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, log3 = mogEM(train3, K, iters,minVary)

    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    ###########################################################################
    #                               Test error                                #
    ###########################################################################
    classification = np.zeros((inputs_test.shape[1],2))
    #probabilities for the twos
    classification[:,0] = mogLogProb(p2,mu2,vary2,inputs_test)
    #probabilities for threes
    classification[:,1] = mogLogProb(p3,mu3,vary3,inputs_test)
    #taking the maximum probability
    classification  = np.argmax(classification,axis=1)
    errorTest[t] = 1-np.mean(classification==target_test) 
    
    ###########################################################################
    #                               Train error                                #
    ###########################################################################
    classification = np.zeros((inputs_train.shape[1],2))
    #probabilities for the twos
    classification[:,0] = mogLogProb(p2,mu2,vary2,inputs_train)
    #probabilities for threes
    classification[:,1] = mogLogProb(p3,mu3,vary3,inputs_train)
    #taking the maximum probability
    classification  = np.argmax(classification,axis=1)
    errorTrain[t] = 1-np.mean(classification==target_train) 

    ###########################################################################
    #                               Valid error                                #
    ###########################################################################
    classification = np.zeros((inputs_valid.shape[1],2))
    #probabilities for the twos
    classification[:,0] = mogLogProb(p2,mu2,vary2,inputs_valid)
    #probabilities for threes
    classification[:,1] = mogLogProb(p3,mu3,vary3,inputs_valid)
    #taking the maximum probability
    classification  = np.argmax(classification,axis=1)
    errorValidation[t] = 1-np.mean(classification==target_valid) 
  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
  plt.plot(numComponents, errorTest,label='test')
  plt.plot(numComponents, errorTrain,label='train')
  plt.plot(numComponents, errorValidation,label='valid')
  plt.legend(loc='best')
  

  plt.draw()
  input('Press Enter to continue.')

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------

  input('Press Enter to continue.')

if __name__ == '__main__':
  # q3()
  q4()
  # q5()

