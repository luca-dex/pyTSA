from Queue import Empty
from multiprocessing import Process

class DataSampler(Process):
    def __init__(self, queueIN, queueOUT, tmin, tmax, step, colname):
        self.queueIN = queueIN
        self.queueOUT = queueOUT
        self.tmin = tmin
        self.tmax = tmax
        self.step = step
        self.colname = colname
        
        super(DataSampler, self).__init__()

    def run(self):
        while True:
            try:
                k,v = self.queueIN.get()
            except Empty:
                break
            else:
                sampled = self.get(v[self.colname], self.tmin, self.tmax, self.step)
                self.queueOUT.put((k, sampled))
                self.queueIN.task_done()


    @staticmethod
    def get(df, l_limit, h_limit, step):
        start = float(l_limit)
        now = float(start + step)
        to_return = []
        try:
            last_value = df.truncate(after=start).tail(1).values[0]
        except:
            last_value = 0
        to_return.append(last_value)
        while now < h_limit:
            try:
                last_value = df.truncate(before=start, after=now).tail(1).values[0]
            except:
                pass
            to_return.append(last_value)
            start = start + step
            now = now + step
        return to_return
