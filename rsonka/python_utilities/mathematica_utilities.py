tableEx = """{{1, x, x^2, x^3, y, y^3, x ** y, y ** x}, {x, x^2, x^3, 1, y ** x, 
  x ** y, y, y^3}, {x^2, x^3, 1, x, y^3, y, y ** x, x ** y}, {x^3, 1, 
  x, x^2, x ** y, y ** x, y^3, y}, {y, x ** y, y^3, y ** x, x^2, 1, 
  x^3, x}, {y^3, y ** x, y, x ** y, 1, x^2, x, x^3}, {x ** y, y^3, 
  y ** x, y, x, x^3, x^2, 1}, {y ** x, y, x ** y, y^3, x^3, x, 1, 
  x^2}}"""

braketEx = "1/Sqrt[2](|0> + |1>)<0| + 1/Sqrt[2](|0> - |1>)<1| + |aXa|"

import re


# Sadly, this doesn't work as well as I would have hoped.
def braFix(matchObj):
    return "Bra[" + matchObj.group(1) + "]"
def ketFix(matchObj):
    return "Ket[" + matchObj.group(1) + "]"

def braKetToMathematica(braKetString):
    toReturn = re.subn("<(.*?)\|", braFix, braKetString)[0] 
    return re.subn("\|(.*?)>", ketFix, toReturn)[0] 
    
        
        
        

'''
def MathematicaTableToPython(tblStr):
    tbl = []
    while re.match
    splitLev = tblStr.split("},")
    print splitLev
    
'''


    
