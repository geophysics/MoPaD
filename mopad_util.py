# -*- coding: utf-8 -*-

import numpy as N
import math

epsilon = 1e-13

class MTError(Exception):
    pass

def die(message):
    sys.stderr.write('Error: %s' % message)
    sys.exit(-1)
    

def cvec(x,y,z):
    """
    Builds a column vector (matrix type) from a 3 tuple.
    """
    return N.matrix( [[x,y,z]], dtype=N.float ).T
    

def strikediprake_2_moments(strike,dip,rake):
    """
    angles are defined as in Jost&Herman (given in degrees)

    strike: angle clockwise between north and plane ( in [0,360[ )
    dip:    angle between surface and dipping plane ( in [0,90] ) 0 = horizontal, 90 = vertical
    rake:   angle on the rupture plane between strike vector and actual movement (defined mathematically positive: ccw rotation is positive)  

    basis for output is NED (= X,Y,Z)

    output:
    
    M = M_nn, M_ee, M_dd, M_ne, M_nd, M_ed
    
    """

    
    S_rad = strike / rad2deg
    D_rad = dip    / rad2deg
    R_rad = rake   / rad2deg

    for ang in S_rad,D_rad,R_rad:
        if abs(ang) < epsilon:
            ang = 0.


    M1 = - ( N.sin(D_rad)*N.cos(R_rad)*N.sin(2*S_rad) + N.sin(2*D_rad)*N.sin(R_rad)*N.sin(S_rad)**2 )
    M2 =   ( N.sin(D_rad)*N.cos(R_rad)*N.sin(2*S_rad) - N.sin(2*D_rad)*N.sin(R_rad)*N.cos(S_rad)**2 )
    M3 =   ( N.sin(2*D_rad)*N.sin(R_rad) )
    M4 =   ( N.sin(D_rad)*N.cos(R_rad)*N.cos(2*S_rad) + 0.5*N.sin(2*D_rad)*N.sin(R_rad)*N.sin(2*S_rad) )
    M5 = - ( N.cos(D_rad)*N.cos(R_rad)*N.cos(S_rad)   + N.cos(2*D_rad)*N.sin(R_rad)*N.sin(S_rad) )
    M6 = - ( N.cos(D_rad)*N.cos(R_rad)*N.sin(S_rad)   - N.cos(2*D_rad)*N.sin(R_rad)*N.cos(S_rad))


    Moments = [M1, M2, M3, M4, M5, M6]

    return tuple(Moments)

#-------------------------------------------------------------------





def euler_to_matrix( alpha, beta, gamma ):
    '''Given the euler angles alpha,beta,gamma, create rotation matrix
        
Given coordinate system (x,y,z) and rotated system (xs,ys,zs)
the line of nodes is the intersection between the x-y and the xs-ys
planes.
    alpha is the angle between the z-axis and the zs-axis.
    beta is the angle between the x-axis and the line of nodes.
    gamma is the angle between the line of nodes and the xs-axis.

Usage for moment tensors:
    m_unrot = numpy.matrix([[0,0,-1],[0,0,0],[-1,0,0]])
    rotmat = euler_to_matrix(dip,strike,-rake)
    m = rotmat.T * m_unrot * rotmat'''
    
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)

    mat = num.matrix( [[cb*cg-ca*sb*sg,  sb*cg+ca*cb*sg,  sa*sg],
                       [-cb*sg-ca*sb*cg, -sb*sg+ca*cb*cg, sa*cg],
                       [sa*sb,           -sa*cb,          ca]], dtype=num.float )
    return mat

def matrix_to_euler( rotmat ):    
    '''
    Returns three Euler angles alpha, beta, gamma (in radians) from a rotation matrix.
    '''
    
    ex = cvec(1.,0.,0.)
    ez = cvec(0.,0.,1.)
    exs = rotmat.T * ex
    ezs = rotmat.T * ez
    enodes = N.cross(ez.T,ezs.T).T
    if N.linalg.norm(enodes) < 1e-10:
        enodes = exs
    enodess = rotmat*enodes
    cos_alpha = float((ez.T*ezs))
    if cos_alpha > 1.: cos_alpha = 1.
    if cos_alpha < -1.: cos_alpha = -1.
    alpha = math.arccos(cos_alpha)
    beta  = math.mod( math.arctan2( enodes[1,0], enodes[0,0] ), math.pi*2. )
    gamma = math.mod( -math.arctan2( enodess[1,0], enodess[0,0] ), math.pi*2. )
    
    return unique_euler(alpha,beta,gamma)

def unique_euler( alpha, beta, gamma ):

    '''Uniquify euler angle triplet.
    
    Puts euler angles into ranges compatible with (dip,strike,-rake) in seismology:
    
    alpha (dip)   : [0, pi/2]
    beta (strike) : [0, 2*pi)
    gamma (-rake) : [-pi, pi)
    
    If alpha is near to zero, beta is replaced by beta+gamma and gamma is set to
    zero, to prevent that additional ambiguity.

    If alpha is near to pi/2, beta is put into the range [0,pi).
    '''
    
    
    alpha = N.mod( alpha, 2.0*pi )
    
    if 0.5*pi < alpha and alpha <= pi:
        alpha = pi - alpha
        beta  = beta + pi
        gamma = 2.0*pi - gamma
    elif pi < alpha and alpha <= 1.5*pi:
        alpha = alpha - pi
        gamma = pi - gamma
    elif 1.5*pi < alpha and alpha <= 2.0*pi:
        alpha = 2.0*pi - alpha
        beta  = beta + pi
        gamma = pi + gamma
    
    
    alpha = N.mod( alpha, 2.0*pi )
    beta  = N.mod( beta,  2.0*pi )
    gamma = N.mod( gamma+pi, 2.0*pi )-pi
    
    # If dip is exactly 90 degrees, one is still
    # free to choose between looking at the plane from either side.
    # Choose to look at such that beta is in the range [0,180)
    
    # This should prevent some problems, when dip is close to 90 degrees:
    if abs(alpha - 0.5*pi) < 1e-10: alpha = 0.5*pi
    if abs(beta - pi) < 1e-10: beta = pi
    if abs(beta - 2.*pi) < 1e-10: beta = 0.
    if abs(beta) < 1e-10: beta = 0.
    
    if alpha == 0.5*pi and beta >= pi:
        gamma = - gamma
        beta  = N.mod( beta-pi,  2.0*pi )
        gamma = N.mod( gamma+pi, 2.0*pi )-pi
        assert 0. <= beta < pi
        assert -pi <= gamma < pi
        
    if alpha < 1e-7:
        beta = N.mod(beta + gamma, 2.0*pi)
        gamma = 0.
    
    return (alpha, beta, gamma)


    
def wrap(text, line_length=80):
    '''Paragraph and list-aware wrapping of text.'''
    
    text = text.strip('\n')
    at_lineend = re.compile(r' *\n')
    at_para = re.compile(r'((^|(\n\s*)?\n)(\s+[*] )|\n\s*\n)')
        
    paragraphs =  at_para.split(text)[::5]
    listindents = at_para.split(text)[4::5]
    newlist = at_para.split(text)[3::5]
   
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
    
    return outlines


# ------------------------------------------------------------------------------

def transform(mat_tup_arr_vec, in_basis, out_basis):
    
    if mat_tup_arr_vec is None:
        return None
    
    lo_bases = ['NED','USE','XYZ','NWU'] 
    
    if (in_basis not in lo_bases):
        raise MTError('Basis not available: %s' % in_basis)
    
    if (out_basis not in lo_bases):
        raise MTError('Basis not available: %s' % out_basis)
    
    if in_basis == out_basis:
        transformed_in = mat_tup_arr_vec

    elif in_basis == 'NED':
        if out_basis=='USE':
            transformed_in = NED2USE(mat_tup_arr_vec)
        if out_basis=='XYZ':
            transformed_in = NED2XYZ(mat_tup_arr_vec)
        if out_basis=='NWU':
            transformed_in = NED2NWU(mat_tup_arr_vec)

    elif in_basis == 'USE':
        if out_basis=='NED':
            transformed_in = USE2NED(mat_tup_arr_vec)
        if out_basis=='XYZ':
            transformed_in = USE2XYZ(mat_tup_arr_vec)
        if out_basis=='NWU':
            transformed_in = USE2NWU(mat_tup_arr_vec)
            
    elif in_basis == 'XYZ':
        if out_basis=='NED':
            transformed_in = XYZ2NED(mat_tup_arr_vec)
        if out_basis=='USE':
            transformed_in = XYZ2USE(mat_tup_arr_vec)
        if out_basis=='NWU':
            transformed_in = XYZ2NWU(mat_tup_arr_vec)

    elif in_basis == 'NWU':
        if out_basis=='NED':
            transformed_in = NWU2NED(mat_tup_arr_vec)
        if out_basis=='USE': 
            transformed_in = NWU2USE(mat_tup_arr_vec)
        if out_basis=='XYZ': 
            transformed_in = NWU2XYZ(mat_tup_arr_vec)

    if len(mat_tup_arr_vec) == 3 and N.prod(N.shape(mat_tup_arr_vec))!=9 :
        tmp_array    = N.array([0,0,0])
        tmp_array[:] =  transformed_in
        return tmp_array
    else:
        return transformed_in

#---------------------------------------------------------------

def _return_matrix_vector_array(ma_ve_ar,basis_change_matrix):

    """
    Generates the output for the functions, yielding matrices, vectors, and arrays in new basis systems.

    Allowed input are 3x3 matrices, 3-vectors, 3-vector collections,
    3-arrays, and 6-tuples.  Matrices are transformed directly,
    3-vectors the same.

    6-arrays are interpreted as 6 independent components of a moment
    tensor, so they are brought into symmetric 3x3 matrix form. This
    is transformed, and the 6 standard components 11,22,33,12,13,23
    are returned.
    """

    
    if (not N.prod(N.shape(ma_ve_ar)) in [3,6,9]) or (not len(N.shape(ma_ve_ar)) in [1,2]):
        raise MTError('wrong input - provide either 3x3 matrix or 3-element vector')

    if  N.prod(N.shape(ma_ve_ar)) == 9:

        return  N.dot(  basis_change_matrix, N.dot(ma_ve_ar, basis_change_matrix.T))

    elif N.prod(N.shape(ma_ve_ar)) == 6:
        m_in        = ma_ve_ar
        orig_matrix = N.matrix( [ [m_in[0],m_in[3],m_in[4]],[ m_in[3],m_in[1],m_in[5] ],[m_in[4],m_in[5],m_in[2]]], dtype=N.float )
        m_out_mat   = N.dot(  basis_change_matrix, N.dot(orig_matrix, basis_change_matrix.T))

        return m_out_mat[0,0],m_out_mat[1,1],m_out_mat[2,2],m_out_mat[0,1],m_out_mat[0,2],m_out_mat[1,2]

    else:
        if N.shape(ma_ve_ar)[0] == 1:
            return  N.dot(basis_change_matrix,ma_ve_ar.transpose())
        else:
            return  N.dot(basis_change_matrix,ma_ve_ar)



#---------------------------------------------------------------

def USE2NED(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    """

    basis_change_matrix = N.matrix( [[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------

def XYZ2NED(some_matrix_or_vector):
    """
    Function for basis transform from basis XYZ to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    """
    
    basis_change_matrix = N.matrix( [[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)
#---------------------------------------------------------------

def NWU2NED(some_matrix_or_vector):
    """
    Function for basis transform from basis NWU to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NWU basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    """
    
    basis_change_matrix = N.matrix( [[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------

def NED2USE(some_matrix_or_vector):
    """
    Function for basis transform from basis  NED to USE.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    """

    basis_change_matrix = N.matrix( [[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------

def XYZ2USE(some_matrix_or_vector):
    """
    Function for basis transform from basis XYZ to USE.

    Input:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    """

    basis_change_matrix = N.matrix( [[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------
def NED2XYZ(some_matrix_or_vector):
    """
    Function for basis transform from basis NED to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------

def NED2NWU(some_matrix_or_vector):
    """
    Function for basis transform from basis NED to NWU.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NWU basis representation

    """
    
    basis_change_matrix = N.matrix( [[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)
#---------------------------------------------------------------

def USE2XYZ(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,0.,1.],[0.,-1.,0.],[1.,0.,0.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)
#---------------------------------------------------------------

def NWU2XYZ(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)
#---------------------------------------------------------------

def NWU2USE(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]], dtype=N.float )

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------
def XYZ2NWU(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)

#---------------------------------------------------------------
def USE2NWU(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis representation

    """

    basis_change_matrix = N.matrix( [[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]], dtype=N.float ).I

    return _return_matrix_vector_array(some_matrix_or_vector,basis_change_matrix)



def fancy_matrix(m_in):
    """

    Returns a given 3x3 matrix or array in a cute way on the shell, if you use 'print' on the return value.
    
    """
    m = m_in.copy()
    
    #    aftercom   = 1 
    #    maxlen =  (int(N.log10(N.max(N.abs(m)))))
    #     if maxlen < 0:
    #         aftercom = -maxlen + 1
    #         maxlen   = 1

    norm_factor = round(max(abs(N.array(m).flatten())),5)
    
    try:
        if  (norm_factor < 0.1) or ( norm_factor >= 10):
            if not abs(norm_factor) == 0:
                m = m/norm_factor
  
                return "\n  / %5.2F %5.2F %5.2F \\\n" % (m[0,0], m[0,1], m[0,2]) +\
                       "  | %5.2F %5.2F %5.2F  |   x  %.2e\n"  % (m[1,0], m[1,1], m[1,2], norm_factor) +\
                       "  \\ %5.2F %5.2F %5.2F /\n" % (m[2,0] ,m[2,1], m[2,2])            
    except:
        pass
           
    
    
    return "\n  / %5.2F %5.2F %5.2F \\\n" % (m[0,0], m[0,1], m[0,2]) +\
           "  | %5.2F %5.2F %5.2F | \n"  % (m[1,0], m[1,1], m[1,2]) +\
           "  \\ %5.2F %5.2F %5.2F /\n" % (m[2,0] ,m[2,1], m[2,2])

    
#-------------------------------------------------------------------

def fancy_vector(v):
    """

    Returns a given 3-vector or array in a cute way on the shell, if you use 'print' on the return value.
    
    """
    return "\n  / %5.2F \\\n" % (v[0]) +\
    "  | %5.2F  |\n"  % (v[1]) +\
    "  \\ %5.2F /\n" % (v[2])

