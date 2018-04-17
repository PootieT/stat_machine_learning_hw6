import numpy as np

def estimate_gaussian(X):
    
    """
    Estimate the mean and standard deviation of a numpy matrix X on a column by column basis
    """
    mu = np.zeros((X.shape[1],))
    var = np.zeros((X.shape[1],))
    ####################################################################
    #               YOUR CODE HERE                                     #
    ####################################################################
    mu = np.mean(X,axis=0)
    var = np.var(X,axis=0,ddof=0)
    print mu.shape, var.shape
    ####################################################################
    #               END YOUR CODE                                      #
    ####################################################################
    return mu, var


def select_threshold(yval,pval):
    """
    select_threshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    bestF1 = 0
    stepsize = (max(pval)-min(pval))/1000
    for epsilon in np.arange(min(pval)+stepsize, max(pval), stepsize):
        
        ####################################################################
        #                 YOUR CODE HERE                                   #
        ####################################################################
        tp = np.sum(np.equal(yval.squeeze(),(pval.T>epsilon)*1.0)*1.0)
        tn = np.sum(np.equal(yval.squeeze(),(pval.T<epsilon)*1.0)*1.0)
        fp = np.sum(yval[(yval.squeeze()==0.0) & (pval.T>epsilon)])
        fn = np.sum(yval[(yval.squeeze()==1.0) & (pval.T<epsilon)])
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = (2.0*prec*rec)/(prec+rec)
        # print "epsilon",epsilon
        # print "yval",yval[:10].squeeze()
        # print "pval",pval[:10],pval[:10].shape
        # print "middle",np.greater(pval,epsilon)
        # print "tp",tp[:10]
        if F1 > bestF1:
            best_epsilon = epsilon
            bestF1 = F1
        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
