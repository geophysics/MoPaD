
import re, sys

def wrap(data, line_length=80):
    data = data.strip('\n')
    at_lineend = re.compile(r' *\n')
    at_para = re.compile(r'((^|(\n\s*)?\n)(\s+[*] )|\n\s*\n)')
        
    paragraphs =  at_para.split(data)[::5]
    listindents = at_para.split(data)[4::5]
    newlist = at_para.split(data)[3::5]
   
    listindents[0:0] = [None]
    listindents.append(True)
    newlist.append(None)
  
    det_indent = re.compile(r'^ *')
    
    iso_latin_1_enc_failed = False
    outlines = []
    for ip, p in enumerate(paragraphs):
        if not p:
            continue
        
        if listindents[ip] is None:
            _indent = det_indent.findall(p)[0]
            findent = _indent
        else:
            findent = listindents[ip]
            _indent = ' '* len(findent)
        
        ll = line_length - len(_indent)
        llf = ll
        
        oldlines = [ s.strip() for s in at_lineend.split(p.rstrip()) ]
        p1 = ' '.join( oldlines )
        possible = re.compile(r'(^.{1,%i}|.{1,%i})( |$)' % (llf, ll))
        for imatch, match in enumerate(possible.finditer(p1)):
            parout = match.group(1)
            if imatch == 0:
                outlines.append(findent + parout)
            else:
                outlines.append(_indent + parout)
            
        if ip != len(paragraphs)-1 and (listindents[ip] is None or newlist[ip] is not None or listindents[ip+1] is None):
            outlines.append('')
    
    return '\n'.join(outlines)
    
    
t = '''
This is a test text.
This is a test text.
This is a test text.
This is a test text.
This is a test text.


        This is a test text.
        This is a test text.
        This is a test text.
        This is a test text.
        This is a test text.
    
    * This is a test text.
      * This is a test text.
        * This is a test text.
          * This is a test text.
    
    
    
        * This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.
          This is a test text.          
        * This is a test text.
        * This is a test text.
        * This is a test text.

Hallo abc

hallo abc
'''


print wrap(t)

