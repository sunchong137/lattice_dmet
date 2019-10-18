
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

def print_summary(flags, summary):

    #frameu = "="*40 + " SUMMARY OF THE QUANTATIES " + "="*40
    #framed = "="*len(frameu)

    #print frameu
    header = " " + flags[0]
    nterm = len(flags)
    for i in xrange(1,nterm):
        header += " "*(19-(len(flags[i])+len(flags[i-1]))/2) + flags[i]
    print header
    nitr = len(summary)
    for itr in xrange(nitr):
        line = "   %d  "%summary[itr][0]
        for i in xrange(1,nterm):
            line += " "*8+"%0.10f"%summary[itr][i] 
        print line
    #print framed


if __name__ == "__main__":
    flags = ["Iter", "Nelec", "Energy", "dE", "dRDM", "dUMAT"]
    summary = [[0,      -0.210210002640 ,         -2.102e-01,       0.070261152339,       1.000000000000,          7.15948e-02 ]]
    summary.append([0,      -0.210210002640 ,         -2.102e-01,       0.070261152339,       1.000000000000,          7.15948e-02 ])
    print_summary(flags, summary)
        
    
    
