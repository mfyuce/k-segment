import numpy as np
import BalancedPartition
import BiCritetria

def Coreset(P, k, eps):
    h = BiCritetria.bicriteria(P, k)
    b = (eps**2 * h) / (100*k*np.log2(P.shape[0]))
    return BalancedPartition.BalancedPartition(P, eps, b)
