"""
Rita Sonka
prettyPrintTable
This is old...it's 2.8 with parenthesis added to print statements. ANd some exceptions. 
basically I tried to load it in python 3 and went back and fixed all the syntax errors. 
"""
#import numpy as np
import textwrap, re
#from ritas_python_util_main import is_float as is_number


def is_number(num_string):
    try:
        float(num_string)
        if float(num_string) == 0:
            return True
        #if float(False) == 0, give that a special exception
        if num_string == False: # wait...shouldn't that be True, for 0?
            return False
        # The below handles np.nan and other things like it, hopefully. 
        if (float(num_string) <=0) ^ (0 < float(num_string)):
            return True
        else:
            return False
    except: # ValueError
        return False

#maxCellSize = 110 # This should probably be a global setting. It's really only necessary for comments right now. 110 because that's the max length of the tables ABOVE the comment table.

# now set as defaults on the two functions that use it.


# Utillity printing functions.
def uPrint(string):
    # Expects a utf8 encoded string. Can fail with some characters, but doesn't crash the program when it does so.
    #print(str(string).encode('utf8', errors='replace')) # I should fix this better at some point.
    print(str(string)) # seems to fix it for python 3?

def preserveDoubleReturnsWrapTo(text, num):
    textParts = text.split("\n\n")
    wrappedText = []
    for part in textParts:
        wrappedText.append(wrapToNum(part, num))
    return "\n\n".join(wrappedText)

def wrapToNum(text, num):
    # Warning: Ignores newlines!
    return textwrap.fill(' '.join(text.split()), num)   

def wrapTo80(text):
    # Warning: Ignores newlines!
    return textwrap.fill(' '.join(text.split()), 80)    

# For use with the below makePrettyTable string, for if I want to do this after having decided on a pretty print::
def convertTableToTabDelimited(tableString):
    class1 = re.sub("[ ]+\|", "\t", tableString) 
    class2 = class1.replace("\n|", "\n")
    class3 = class2[1:].replace("|", "\t")    
    return class3

def printTabDelimitedTable(table):
    print(convertTableToTabDelimited(makePrettyTableString(table)))




testRoundToString = [1, 1.2, 1.234567, ]


swapToE = 6 # When to do exponential notation? See below
# This function needs work.
debugPrint = 0
def debugPrint(string):
    if debugPrint == 1:
        print(string)
        
def roundToString(num, rTo):
    '''Helper method for roundTableContentsToSigFig'''
    nS = str(num)
    if nS[-3:] == 'inf':
        return nS # let's just hope this is okay.
    order = 0 # What power of ten to multiply num by to get actual num ; 
    neg = ''
    if nS[0] == '-':
        neg = '-'
        nS = nS[1:]
    # used to relocate '.' and decide on whether to use exponential notation
    eLoc = nS.find('e')
    if not eLoc == -1:
        order += float(nS[eLoc+1:]) # Yeah, that works. Python's amazing.
        nS = nS[:eLoc]
    if not "." in nS: # Yes this needs to come after the above.
        nS += "."
    if nS[0] == '0':
        nS = nS[1:]
    [preDec, postDec] = nS.split(".")
    order -= len(postDec)
    if preDec == '':
        postDec = postDec.strip("0")
    nS = preDec+postDec
    
    if len(nS) > rTo: # More digits than we want.
        start = nS[:rTo] # Note we'll  need one more digit than this to do the rounding.
        order += len(nS[rTo:])
        #print str((nS, start, str(round(int(nS[:rTo+1]), -1))))      
        nS = str(round(int(nS[:rTo+1]), -1))[:-3] #round adds a .0 on. 
        '''
        if int(nS[rTo]) >= 5:
            nS = nS[:rTo-1] + str(int(nS[rTo-1])+1)
        else:
            nS = nS[:rTo]  '''
            
    # Alright. The irrelevant sig figs have been pruned. Add back in?
    if rTo > swapToE or \
       (order>=0 and order+len(nS) > swapToE) or \
       (order<0 and max(abs(order), len(nS))>swapToE):
        # Let's do exponential notation instead of normal display.
        order += len(nS)-1
        nS = nS[0]+ "."+ nS[1:] + "e"
        if order >= 0:
            nS += "+"
        nS += str(int(order))
        debugPrint("returning:"+neg+nS + " original:" +str(num))
        return neg + nS
    # Not using exponential notation
    if order >= 0: # Easy case...
        debugPrint("returning:"+neg+nS+'0'*order + " original:" +str(num))
        return neg + nS + '0'*order
    # Order is < 0
    if abs(order) >= len(nS):
        debugPrint("returning:"'0.' + '0'*(abs(len(nS) + int(order))) + nS +" original:" +str(num))
        return '0.' + '0'*(abs(len(nS) + int(order))) + nS # Remember order <0
    # Order < 0 and len(nS) > order
    debugPrint("returning:"+neg + nS[:order] + "." + nS[order:] + " original:" +str(num))
    return neg + nS[:order] + "." + nS[order:]
        

def isiterable(obj):
    # Maybe someday use this to have them not require 2d tables.
    try:
        object_iterator = iter(obj)
        return True
    except (TypeError, te):
        return False  

def roundDictToSF(dicty, rTo):
    dic2 = {}
    for key in dicty:
        if is_number(dicty[key]):
            dic2[key] = roundToString(dicty[key], rTo)
        else:
            dic2[key] = dicty[key]
    return dic2
            

# NOte you'll have to import numpy as np...
#(table0, table1, table2, table3) = ("11.11", ["11.11", 22.22], [["11.11", 22.22],np.array([33, 44.4])], [[["11.11", 22.22], np.array([33, 44.4])], [[55, 666], ["sevens", {0:"88",32:"99.9", 2:"111"}]]])
def recursiveTableRoundSigFig(table, rTo):
    '''Returns a 'copy' of table with all the numbers replaced by string 
    representations of those numbers, rounded to significant figure rTo.
    Does not change the originial table, because numpy will just turn the string back into a number.
    May necessitate use of scientific notation. 
    rTo > 10 will cause issues due to how it uses Python's str() display.'''
    # Base case: 
    tbl = []
    if is_number(table):
        return roundToString(table, rTo)
    elif isinstance(table, str): # if string, we're really just done with this branch.
        return table 
    # I should really just figure out how to identify numpy arrays and add them in.
    elif isiterable(table) and not isinstance(table, str):  
        try:
            for i in range(len(table)): 
                #table[i] = recursiveTableRoundSigFig(table[i], rTo)
                tbl.append(recursiveTableRoundSigFig(table[i], rTo))
        except: # Note this can still hit dictionaries with the right keys...
            return table # That's not really a table if above doesn't work. Return what it is.
    return tbl


def roundTableContentsToSigFig(table, rTo):
    '''Returns a python list copy of table with all the numbers replaced by string 
    representations of those numbers, rounded to significant figure rTo.
    Does not change the originial table, because numpy will just turn the string back into a number.
    May necessitate use of scientific notation. 
    rTo > 10 will cause issues due to how it uses Python's str() display.'''
    # Now makes a new table,  
    # Yes, seriously.
    tbl = []
    for rI in range(len(table)):
        tbl.append([])
        for cI in range(len(table[rI])):
            val = table[rI][cI]
            if is_number(val):
                tbl[rI].append(roundToString(val, rTo))
            else:
                tbl[rI].append(val)
    return tbl
    
def copyWithElementsStrings(table):
    # You really should be giving nicer input than this, but just in case...
    return [[str(table[r][c]) for c in range(len(table[r]))] for r in range(len(table))]



def makePrettyTableString(origTable, maxCellSize=110):
    # assumes a rectangular table in the form of a list of rows that are themselves lists of columns
    # e.g. [[row1col1, row1col2], [row2col1, row2col2]]
    # also assumes no \n or \r in elements. I SHOULD FIX THIS, 
    # IT LOOKS TERRIBLE ON MONITOR BECAUSE it has extra whitespace. 
    # I think because it's counting the full string for length and setting columns
    # accordingly.
    if len(origTable) < 1:
        print("Bad table: " + str(origTable))
        return -1
    """elif len(origTable[0]) < 1:
        print("Bad table: " + str(origTable))
        return -1"""
    # Try to sanitize any bad input a little.
    table = copyWithElementsStrings(origTable)
    
    # Figuring out how to space the table. =======
    # Find number of columns in longest row.
    colSizes = [] # How many characters in each column?
    rowHeights = [] # How many lines in each row of the table? 
    numCols = len(table[0])
    for row in table:
        if numCols < len(row):
            numCols = len(row)
        rowHeights.append(0)
    for col in range(numCols):
        colSizes.append(0)
    # Find maximum size of each column. Now supporting newlines.
    #, up to maxCellSize (which possibly should be a setting?) Have decided to trust input for now.
    for rowNum in range(len(table)):
        row = table[rowNum]
        for colNum in range(len(row)):
            #text = str(row[colNum]) # Why was this commented out?
            text = wrapToNum(str(row[colNum]), maxCellSize)
            lines = text.splitlines() # splitlines recognizes \r\n
            row[colNum] = lines[:] # Note that I save this back! It saves a lot of splitlining.
            if len(lines) > rowHeights[rowNum]:
                rowHeights[rowNum] = len(lines)
            for line in lines:
                if len(line) > maxCellSize:
                    print("Line size issue: ")
                    uPrint(line)
                if len(line) > colSizes[colNum]:
                    colSizes[colNum] = len(line)  
            #if isinstance(row[colNum], basestring):
            #    if len(row[colNum]) > colSizes[colNum]:
            #        colSizes[colNum] = len(row[colNum])
            #elif len(str(row[colNum])) > colSizes[colNum]:
            #    colSizes[colNum] = len(str(row[colNum]))
            
            
            #if colSizes[colNum] > maxCellSize:
            #    colSizes[colNum] = maxCellSize
    # Make the table!
    tblString = ""
    for rowNum in range(len(table)):
        row = table[rowNum]
        #rowString = "|"
        if rowHeights[rowNum] > 1 or (rowNum > 0 and rowHeights[rowNum-1] > 1):
            # It's possible that I should leave this to the input, and not force it in.  Also possible that 
            tblString += "|" + (sum(colSizes) + len(colSizes)-1)*"-" + "|\n" # Note len(colSizes) and sum(colSizes) can't be zero if rowHeights[rowNum]>1.
        for lineNum in range(rowHeights[rowNum]):
            lineString = "|"   
            for colNum in range(len(row)):
                lines = row[colNum]# Remember the splitlined result was saved
                if len(lines) > lineNum: 
                    item = lines[lineNum]
                else:
                    item = ""
                lineString += item
                lineString += ' ' * (colSizes[colNum] - len(item))
                lineString += '|'
                
                """
                if isinstance(row[colNum], basestring):
                    if len(table[0]) > 0 and not re.match('-----BELOW.*', table[0][0]): # A WAY OF IDENTIFYING COMMENTS  # .encode('ascii','replace')
                        item = row[colNum].replace('\n', '\\n').replace('\r', '\\r')
                    else:
                        item = row[colNum]
                else:
                    item = str(row[colNum])#.replace('\n', '\\n').replace('\r', '\\r')
                rowString += item
                rowString += ' ' * (colSizes[colNum] - len(item))
                rowString += '|'
                """
            tblString += lineString + "\n"
    #print colSizes
    return tblString    
    
def prettyPrintTable(table, maxCellSize=110):
    # assumes a rectangular table in the form of a list of rows that are themselves lists of columns
    # e.g. [[row1col1, row1col2], [row2col1, row2col2]]
    # also assumes no \n in elements
    uPrint(makePrettyTableString(table, maxCellSize=maxCellSize))