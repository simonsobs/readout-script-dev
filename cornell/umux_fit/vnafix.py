import matplotlib.pyplot as plt;
from numpy import *;
import sys;

def hpfilter(data, a):
    output = data[:];
    for i in range(1,len(output)):
        output[i] = a*output[i-1] + a*(data[i] - data[i-1]);
    return output;

def find_resonances(data):
    filtered = hpfilter(data,0.5);
    output = [];
    lowrange = None;
    hirange = None;
    below = False; #Are we below 0.5?
    above = False; #Have we gone below 0.5 and also above 0 before finding a peak?
    for i in range(10,len(filtered)):
        if not below:
            if filtered[i]<-0.2:
                lowrange=i;
                below=True;
                above=False;
            continue;
        #Now below must be true.
        elif not above:
            if filtered[i]>0.0:
                above=True;
            continue;
        #Now both must be true, so we are looking for a peak
        elif filtered[i]<filtered[i-1]: #A drop
            hirange = i;
            output.append(argmin(data[lowrange:hirange])+lowrange);
            #Reset
            below = False;
            above = False;
    return output;

#f = open('vnasweep2.txt');

if len(sys.argv)<2:
    print "Pass input file";
    exit();
f = open(sys.argv[1]);

freqs = [];
dBs = [];
for l in f:
    a = l.split();
    freqs.append(float(a[0]));
    dBs.append(float(a[1]));
plt.plot(freqs,dBs);
plt.plot(freqs[10:],hpfilter(dBs,0.5)[10:]);
plt.axhline(-0.5);
res = find_resonances(dBs);
print "#",len(res);
plt.plot([freqs[i] for i in res],[dBs[i] for i in res],"o");
plt.show();
plt.clf();

for r in res:
    print str(freqs[r])+"\t"+str(dBs[r]);
