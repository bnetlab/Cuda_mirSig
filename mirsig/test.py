from math import sqrt
from collections import defaultdict
import joblib

class BatchCompletionCallBack(object):
  completed = defaultdict(int)
  total = 0
  def __init__(self, time, index, parallel):
    self.index = index
    self.parallel = parallel

  def __call__(self, index):
    BatchCompletionCallBack.completed[self.parallel] += 1
    print("done with {}".format(self.total*BatchCompletionCallBack.completed[self.parallel]))
    if self.parallel._original_iterator is not None:
      self.parallel.dispatch_next()
BatchCompletionCallBack.total = 10
joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack

if __name__ == "__main__":
    print(joblib.Parallel(n_jobs=2)(joblib.delayed(sqrt)(i**2) for i in range(10)))