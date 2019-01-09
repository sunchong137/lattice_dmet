
class Logger(object):
    def __init__(self,fname):
#        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
#        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
	self.log.flush()
        pass    
