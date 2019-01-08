Instructions for uMux resonator fitting code
by Jason Stevens, jrs584@cornell.edu on Jan 7, 2019

These python scripts are based on arxiv:1410.3365,
"Efficient and robust analysis of complex scattering data under noise in microwave resonators."

This code is designed for our specific application at Cornell but could be modified as desired for your purposes.
Specifically, you may want to consider the format of the input and the desired output.

Input data files are generally of the following format:
There are three columns, whitespace separated, which contain a VNA trace.
Column 0 has frequency (x-axis) in Hertz.
Column 1 has decibels.
Column 2 has phase, in degrees. The code will automatically unwrap the phase as necessary.

To use these scripts, you should know your system's "cable delay," called "tau". This parameter is
described in the aforementioned paper but should be roughly the amount of time in seconds it takes for a signal
to pass between the two VNA ports through the device under test. For our setup at Cornell, I use a value
of 7e-8, which is the default provided here. In my experience, the analysis is not very sensitive to
this number, so if it is off by some order one factor, it should still give approximately the same
results. I have not included the script I use to find this number, since it usually requires some
playing around with to get the right results. If interest is expressed, I can develop this code further
and add it to the repository. Note, as of today, there is still some residual code in the scripts to
find tau, but it is not executed; I intend to remove this. Do not trust it as-is.
The number is hard-coded in each of the provided scripts. All hard coded parameters that you may want to
adjust, including cable delay, will always be at the top of the script, under the import statements.
Change this value to the appropriate value for your system, and you may not have to change it again.

There are currently two scripts. The first is find_f0.py and is the more basic of the two.
When passed a data file as a command line parameter, it will output the center resonance frequency.
The heavy lifting is done by the function fitDataRes0, which takes as input parameters the file name and
tau.
While this code does not output Q, it does calculate the Q total internally. You can modify fitDataRes0 to
output total Q by returning params[2] at the end of the function. This assumes a good fit.
If the fit does not work, the function will return the initial guess for f0 instead. This is just the
minimum of the trace.

superfit.py is designed to produce SQUID f-phi curves using the same code as find_f0.py. It takes as input
an entire directory full of data files. The filenames are expected to be in the format X.XvN.txt where
X.X is the flux ramp line voltage and N is the resonator number. It will output to the subdirectory plots/
two files per resonator. The first is a plot of the f0 vs flux ramp voltage, and the second file is a
text file containing the data of the plot.

I have included a sample data file, 0.0v2.txt.

BONUS:
I'm also going to throw in my vnafix.py script. This takes as input a vna trace data file (like described above)
that contains many (perhaps hundreds) of resonators, instead of just one. It will, roughly, find the center
frequency of each of them and output a list. There is relatively little sophistication here, though it does
use a high pass filter on the trace. This is how I generate a list of resonators to feed to the data acquisition
code. I am not currently uploading the data acquisition code because it is specialized to our system and
probably won't work on others without heavy modification.
