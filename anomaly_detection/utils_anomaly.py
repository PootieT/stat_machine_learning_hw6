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
        pred = (pval.T>epsilon)*1
        # tp = np.sum(np.equal(yval,pred)*1.0)
        tp = np.sum(((yval)&(pred))*1.0)
        fp = np.sum(np.equal(yval,1-pred)*1.0)
        fn = np.sum(((yval==1) & (1-pred))*1.0)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = (2.0*prec*rec)/(prec+rec)
        # print "precision:",prec,"recall:",rec,"F1:",F1

        # # print "epsilon",epsilon
        # print "yval",yval[:20].squeeze().T
        # # # print "pval",pval[80:100]
        # print "pred",pred[:20]
        # print "true_positive", (((yval==1)&(pred))*1.0)[:20],"and ",np.sum((((yval)&(pred))*1.0)[:20])
        # print "fals_positive", (np.equal(yval.squeeze(),1-pred)*1.0)[:20],"and ",np.sum((np.equal(yval.squeeze(),1-pred)*1.0)[:20])
        # print "fals_negative", ((yval.squeeze()==1) & (1-pred)).squeeze().T[:20], "and ",np.sum(((yval.squeeze()==1) & (1-pred)).squeeze().T[:20])
        # print "precision: ",prec, "recall: ",rec
        # print "epislon",epsilon,"F1",F1
        # print " "
        if F1 > bestF1:
            best_epsilon = epsilon
            bestF1 = F1
            print "best epislon",best_epsilon,"best F1",bestF1
        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
