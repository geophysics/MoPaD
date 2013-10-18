#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

#######################################################################
#########################   MoPaD  ####################################

######### Moment tensor Plotting and Decomposition tool #############
#######################################################################

Multi method tool for:

- Plotting and saving of focal sphere diagrams ('Beachballs').

- Decomposition and Conversion of seismic moment tensors.

- Generating coordinates, describing a focal sphere diagram, to be
piped into GMT's psxy (Useful where psmeca or pscoupe fail.)

#######################################################################

Version  0.9b

THIS IS AN ALPHA VERSION -- FOR TESTING PURPOSES ONLY

THIS SOFTWARE COMES WITH NO WARRANTY

#######################################################################

Copyright (C) 2010
Lars Krieger & Sebastian Heimann

Contact
lars.krieger@zmaw.de  &  sebastian.heimann@zmaw.de

#######################################################################

License:

GNU Lesser General Public License, Version 3

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.

"""


#############################################################

mopad_version = 0.9


#standard libraries:
import sys, optparse, re, math
from cStringIO import StringIO


#additional library:
import numpy as N


#constants:
dynecm = 1e-7
pi = N.pi

epsilon = 1e-13

rad2deg = 180./pi

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

def basis_switcher(in_system, out_system):
    from_ned = {
        'NED': N.matrix( [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=N.float ),
        'USE': N.matrix( [[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]], dtype=N.float ).I,
        'XYZ': N.matrix( [[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]], dtype=N.float ).I,
        'NWU': N.matrix( [[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=N.float ).I }

    return from_ned[in_system].I * from_ned[out_system]

def basis_transform_matrix(m, in_system, out_system):
    r = basis_switcher(in_system, out_system)
    return N.dot(r, N.dot(m, r.I))

def basis_transform_vector(v, in_system, out_system):
    r = basis_switcher(in_system, out_system)
    return N.dot(r, v)

class MopadHelpFormatter(optparse.IndentedHelpFormatter):

    def format_option(self, option):
        '''From IndentedHelpFormatter but using a different wrap method.'''

        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else:                       # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            help_lines = wrap(help_text, self.help_width)
            if len(help_lines) > 1:
                help_lines.append('')
            result.append("%*s%s\n" % (indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)

    def format_description(self, description):
        if not description:
            return ""
        desc_width = self.width - self.current_indent
        indent = " "*self.current_indent
        return '\n'.join(wrap(description, desc_width)) + "\n"


#------------------------------------------
class MTError(Exception):
    pass
#------------------------------------------


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

    mat = N.matrix( [[cb*cg-ca*sb*sg,  sb*cg+ca*cb*sg,  sa*sg],
                       [-cb*sg-ca*sb*cg, -sb*sg+ca*cb*cg, sa*cg],
                       [sa*sb,           -sa*cb,          ca]], dtype=N.float )
    return mat


class MomentTensor:

    _m_unrot = N.matrix( [[0.,0.,-1.],[0.,0.,0.],[-1.,0.,0.]], dtype=N.float )

    def __init__(self, M=None, in_system='NED', out_system='NED', debug=0):
        """
        Creates a moment tensor object on the basis of a provided mechanism M.

        If M is a non symmetric 3x3-matrix, the upper right triangle
        of the matrix is taken as reference. M is symmetrisised
        w.r.t. these entries. If M is provided as a 3-,4-,6-,7-tuple
        or array, it is converted into a matrix internally according
        to standard conventions (Aki & Richards).

        'system' may be chosen as 'NED','USE','NWU', or 'XYZ'.

        'debug' enables output on the shell at the intermediate steps.

        """


        self._original_M       = M[:]

        self._input_basis = in_system.upper()
        self._output_basis = out_system.upper()

        # bring M to symmetric matrix form
        self._M                 = self._setup_M(M, self._input_basis)

        #decomposition:
        self._decomposition_key = 1

        #eigenvector / principal-axes system:
        self._eigenvalues       = None
        self._eigenvectors      = None
        self._null_axis         = None
        self._t_axis            = None
        self._p_axis            = None
        self._rotation_matrix   = None

        # optional - maybe set afterwards by external application - for later plotting:
        self._best_faultplane   = None
        self._auxiliary_plane   = None

        #initialise decomposition components
        self._DC                 = None
        self._DC_percentage      = None
        self._DC2                = None
        self._DC2_percentage     = None
        self._DC3                = None
        self._DC3_percentage     = None

        self._iso                = None
        self._iso_percentage     = None
        self._devi               = None
        self._devi_percentage    = None
        self._CLVD               = None
        self._CLVD_percentage    = None

        self._isotropic         = None
        self._deviatoric        = None
        self._seismic_moment    = None
        self._moment_magnitude  = None

        self._decomp_attrib_map_keys = ('in','out','type',
        'full',
        'iso','iso_perc',
        'dev','devi','devi_perc',
        'dc','dc_perc',
        'dc2','dc2_perc',
        'dc3','dc3_perc',
        'clvd','clvd_perc',
        'mom','mag',
        'eigvals','eigvecs',
        't','n','p')

        self._decomp_attrib_map = dict(zip( self._decomp_attrib_map_keys,
        ('input_system','output_system','decomp_type',
        'M',
        'iso','iso_percentage',
        'devi','devi','devi_percentage',
        'DC','DC_percentage',
        'DC2','DC2_percentage',
        'DC3','DC3_percentage',
        'CLVD','CLVD_percentage',
        'moment','mag',
        'eigvals','eigvecs',
        't_axis','null_axis','p_axis')
        ))




        #carry out the MT decomposition - results are in basis NED
        self._decompose_M()

        #set the appropriate principal axis system:
        self._M_to_principal_axis_system()

    #---------------------------------------------------------------

    def _setup_M(self,mech, input_basis):
        """
        Brings the provided mechanism into symmetric 3x3 matrix form.


        The source mechanism may be provided in different forms:

        -- as 3x3 matrix - symmetry is checked - one basis system has to be chosen, or NED as default is taken
        -- as 3-element tuple or array - interpreted as strike, dip, slip-rake angles in degree
        -- as 4-element tuple or array - interpreted as strike, dip, slip-rake angles in degree + seismic scalar moment in Nm
        -- as 6-element tuple or array - interpreted as the 6 independent entries of the moment tensor
        -- as 7-element tuple or array - interpreted as the 6 independent entries of the moment tensor + seismic scalar moment in Nm
        -- as 9-element tuple or array - interpreted as the 9 entries of the moment tensor - checked for symmetry
        -- as a nesting of one of the upper types (e.g. a list of n-tuples) - first element of outer nesting is taken

        """

        #set source mechanism to matrix form

        if mech==None:
            raise MTError('Please provide a mechanism')

        # if some stupid nesting occurs
        if len(mech) ==  1:
            mech  = mech[0]

        # all 9 elements are given
        if N.prod(N.shape(mech)) == 9:
            if N.shape(mech)[0] == 3:
                #assure symmetry:
                mech[1,0] = mech[0,1]
                mech[2,0] = mech[0,2]
                mech[2,1] = mech[1,2]
                new_M     = mech
            else:
                new_M     = N.array(mech).reshape(3,3).copy()
                new_M[1,0]= new_M[0,1]
                new_M[2,0]= new_M[0,2]
                new_M[2,1]= new_M[1,2]


        # mechanism given as 6- or 7-tuple, list or array
        elif len(mech) == 6 or len(mech) == 7:
            M        = mech
            new_M    = N.matrix( N.array([M[0],M[3],M[4],M[3],M[1],M[5],M[4], M[5],M[2] ]).reshape(3,3) )

            if len(mech) == 7 :
                new_M    = M[6] * new_M

        # if given as strike, dip, rake, conventions from Jost & Herrmann hold - resulting matrix is in NED-basis:
        elif len(mech) == 3 or len(mech) == 4 :
            strike, dip, rake = mech[:3]
            scalar_moment = 1.0
            if len(mech) == 4:
                scalar_moment = mech[3]

            rotmat1 = euler_to_matrix( dip/rad2deg, strike/rad2deg, -rake/rad2deg )
            new_M = rotmat1.T * MomentTensor._m_unrot * rotmat1 * scalar_moment

            #to assure right basis system - others are meaningless, provided these angles
            input_basis   =   'NED'

        return  basis_transform_matrix(N.matrix(new_M), input_basis, 'NED')



    #---------------------------------------------------------------

    def _decompose_M(self):
        """
        Running the decomposition of the moment tensor object.

        the standard decompositions M = Isotropic + DC + (CLVD or 2nd DC) are supported (C.f. Jost & Herrmann, Aki & Richards)

        """
        k = self._decomposition_key
        d = MomentTensor.decomp_dict
        if k in d:
            d[k][1](self)

        else:
            raise MTError('Invalid decomposition key: %i' % k)

    #---------------------------------------------------------------

    def print_decomposition(self):
        for arg in self._decomp_attrib_map_keys:
            print getattr(self,'get_'+self._decomp_attrib_map[arg])(style='y',system=self._output_basis)
            

    #---------------------------------------------------------------


    def _standard_decomposition(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        - isotropic
        - deviatoric
        - DC
        - CLVD

        parts of the input moment tensor.

        results are given as attributes, which can be returned via 'get_<name of attribute>' functions:

        DC
        CLVD
        DC_percentage
        seismic_moment
        moment_magnitude

        """
        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

        self._isotropic  = M_iso
        self._deviatoric = M_devi

        #eigenvalues and -vectors
        eigenwtot,eigenvtot  = N.linalg.eig(M_devi)

        #eigenvalues and -vectors of the deviatoric part
        eigenw1,eigenv1  = N.linalg.eig(M_devi)

        #eigenvalues in ascending order:
        eigenw           = N.real( N.take( eigenw1,N.argsort(abs(eigenwtot)) ) )
        eigenv           = N.real( N.take( eigenv1,N.argsort(abs(eigenwtot)) ,1 ) )

        #eigenvalues in ascending order in absolute value!!:
        eigenw_devi           = N.real( N.take( eigenw1,N.argsort(abs(eigenw1)) ) )
        eigenv_devi           = N.real( N.take( eigenv1,N.argsort(abs(eigenw1)) ,1 ) )

        M0_devi          = max(abs(eigenw_devi))

        #named according to Jost & Herrmann:
        a1 = eigenv[:,0]#/N.linalg.norm(eigenv[:,0])
        a2 = eigenv[:,1]#/N.linalg.norm(eigenv[:,1])
        a3 = eigenv[:,2]#/N.linalg.norm(eigenv[:,2])

        # if only isotropic part exists:
        if M0_devi < epsilon:
            F = 0.5
        else:
            F           = -eigenw_devi[0]/eigenw_devi[2]


        M_DC        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_CLVD      = N.matrix(N.zeros((9),float)).reshape(3,3)

        M_DC        = eigenw[2]*(1-2*F)*( N.outer(a3,a3) - N.outer(a2,a2) )
        M_CLVD      = M_devi - M_DC #eigenw[2]*F*( 2*N.outer(a3,a3) - N.outer(a2,a2) - N.outer(a1,a1))


        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi

        M_iso_percentage     = int(round(M0_iso/M0 *100,6))
        self._iso_percentage = M_iso_percentage


        M_DC_percentage = int(round(( 1 - 2 * abs(F) )* ( 1 - M_iso_percentage/100.)  * 100,6))


        self._DC            =  M_DC
        self._CLVD          =  M_CLVD
        self._DC_percentage =  M_DC_percentage

        #self._seismic_moment   = N.sqrt(1./2*N.sum(eigenw**2) )
        self._seismic_moment   = M0
        self._moment_magnitude = N.log10(self._seismic_moment*1.0e7)/1.5 - 10.7

    #---------------------------------------------------------------
    def _decomposition_w_2DC(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        - isotropic
        - deviatoric
        - 2 DC

        parts of the input moment tensor.

        results are given as attributes, which can be returned via 'get_<name of attribute>' functions:

        DC1
        DC2
        DC_percentage
        seismic_moment
        moment_magnitude

        """
        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

        self._isotropic  = M_iso
        self._deviatoric = M_devi

        #eigenvalues and -vectors of the deviatoric part
        eigenw1,eigenv1  = N.linalg.eig(M_devi)


        #eigenvalues in ascending order of their absolute values:
        eigenw           = N.real( N.take( eigenw1,N.argsort(abs(eigenw1)) ) )
        eigenv           = N.real( N.take( eigenv1,N.argsort(abs(eigenw1)) ,1 ) )

        M0_devi          = max(abs(eigenw))

        #named according to Jost & Herrmann:
        a1 = eigenv[:,0]
        a2 = eigenv[:,1]
        a3 = eigenv[:,2]


        M_DC        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_DC2       = N.matrix(N.zeros((9),float)).reshape(3,3)

        M_DC        = eigenw[2]*( N.outer(a3,a3) - N.outer(a2,a2) )
        M_DC2       = eigenw[0]*( N.outer(a1,a1) - N.outer(a2,a2) )

        M_DC_percentage = abs(eigenw[2]/(abs(eigenw[2])+abs(eigenw[0]) )     )

        self._DC            =  M_DC
        self._DC2           =  M_DC2
        self._DC_percentage =  M_DC_percentage

        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi

        M_iso_percentage     = int(M0_iso/M0 *100)
        self._iso_percentage = M_iso_percentage


        #self._seismic_moment   = N.sqrt(1./2*N.sum(eigenw**2) )
        self._seismic_moment   = M0
        self._moment_magnitude = N.log10(self._seismic_moment*1.0e7)/1.5 - 10.7

    #---------------------------------------------------------------
    def _decomposition_w_CLVD_2DC(self):
        """
        Decomposition according to Dahm (1993) into

        - isotropic
        - CLVD
        - strike-slip
        - dip-slip

        parts of the input moment tensor.

        results are given as attributes, which can be returned via 'get_<name of attribute>' functions:

        iso
        CLVD
        DC1
        DC2
        iso_percentage
        DC_percentage
        DC1_percentage
        DC2_percentage
        CLVD_percentage
        seismic_moment
        moment_magnitude

        """
        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

        self._isotropic  = M_iso
        self._deviatoric = M_devi

        M_DC1        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_DC2        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_CLVD       = N.matrix(N.zeros((9),float)).reshape(3,3)

        M_DC1[0,0]   = -0.5*(M[1,1]-M[0,0])
        M_DC1[1,1]   = 0.5*(M[1,1]-M[0,0])
        M_DC1[0,1]   = M_DC1[1,0]   = M[0,1]

        M_DC2[0,2]   = M_DC2[2,0]   = M[0,2]
        M_DC2[1,2]   = M_DC2[2,1]   = M[1,2]

        M_CLVD       =  1./3.*(0.5*(M[1,1]+M[0,0])-M[2,2])*N.diag( N.array([1.,1.,-2.] ) )

        M_DC         =  M_DC1 + M_DC2

        self._DC            =  M_DC
        self._DC1           =  M_DC1
        self._DC2           =  M_DC2

        self._DC_percentage =  M_DC1_perc
        self._DC2_percentage=  M_DC2_perc

        #according to Bowers & Hudson:
        eigvals_M, dummy_vecs      =  N.linalg.eig(M)
        eigvals_M_devi, dummy_vecs =  N.linalg.eig(M_devi)
        eigvals_M_iso, dummy_iso   =  N.linalg.eig(M_iso)
        eigvals_M_clvd, dummy_vecs =  N.linalg.eig(M_CLVD)
        eigvals_M_dc1, dummy_vecs  =  N.linalg.eig(M_DC1)
        eigvals_M_dc2, dummy_vecs  =  N.linalg.eig(M_DC2)

        #M0_M        = N.max(N.abs(eigvals_M - 1./3*N.sum(eigvals_M)   ))
        M0_M_iso    = N.max(N.abs(eigvals_M_iso - 1./3*N.sum(eigvals_M)   ))
        M0_M_clvd   = N.max(N.abs(eigvals_M_clvd - 1./3*N.sum(eigvals_M)   ))
        M0_M_dc1    = N.max(N.abs(eigvals_M_dc1 - 1./3*N.sum(eigvals_M)   ))
        M0_M_dc2    = N.max(N.abs(eigvals_M_dc2 - 1./3*N.sum(eigvals_M)   ))
        M0_M_dc     = M0_M_dc1 + M0_M_dc2
        M0_M_devi   = M0_M_clvd + M0_M_dc

        M0_M        = M0_M_iso + M0_M_devi

        M_iso_percentage     = int(M0_M_iso/M0_M *100)
        self._iso_percentage = M_iso_percentage

        M_DC_percentage     = int(M0_M_dc/M0_M *100)
        self._dc_percentage = M_dc_percentage
        M_DC1_percentage     = int(M0_M_dc1/M0_M *100)
        self._dc1_percentage = M_dc_percentage
        M_DC2_percentage     = int(M0_M_dc2/M0_M *100)
        self._dc2_percentage = M_dc_percentage



        #self._seismic_moment   = N.sqrt(1./2*N.sum(eigenw**2) )
        self._seismic_moment   = M0_M
        self._moment_magnitude = N.log10(self._seismic_moment*1.0e7)/1.5 - 10.7

    #---------------------------------------------------------------
    def _decomposition_w_3DC(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        - isotropic
        - deviatoric
        - 3 DC

        parts of the input moment tensor.

        results are given as attributes, which can be returned via 'get_<name of attribute>' functions:

        DC1
        DC2
        DC3
        DC_percentage
        seismic_moment
        moment_magnitude

        """
        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

        self._isotropic  = M_iso
        self._deviatoric = M_devi

        #eigenvalues and -vectors of the deviatoric part
        eigenw1,eigenv1  = N.linalg.eig(M_devi)
        M0_devi          = max(abs(eigenw1))

        #eigenvalues and -vectors of the full M !!!!!!!!
        eigenw1,eigenv1  = N.linalg.eig(M)


        #eigenvalues in ascending order of their absolute values:
        eigenw           = N.real( N.take( eigenw1,N.argsort(abs(eigenw1)) ) )
        eigenv           = N.real( N.take( eigenv1,N.argsort(abs(eigenw1)) ,1 ) )


        #named according to Jost & Herrmann:
        a1 = eigenv[:,0]
        a2 = eigenv[:,1]
        a3 = eigenv[:,2]


        M_DC1        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_DC2        = N.matrix(N.zeros((9),float)).reshape(3,3)
        M_DC3        = N.matrix(N.zeros((9),float)).reshape(3,3)

        M_DC1        = 1./3.*(eigenw[0] - eigenw[1]) *( N.outer(a1,a1) - N.outer(a2,a2) )
        M_DC2        = 1./3.*(eigenw[1] - eigenw[2]) *( N.outer(a2,a2) - N.outer(a3,a3) )
        M_DC3        = 1./3.*(eigenw[2] - eigenw[0]) *( N.outer(a3,a3) - N.outer(a1,a1) )

        M_DC1_perc = int(100*abs((eigenw[0]-eigenw[1]))/ (abs((eigenw[1]-eigenw[2]))+abs((eigenw[1]-eigenw[2]))+abs((eigenw[2]-eigenw[0]))))
        M_DC2_perc = int(100*abs((eigenw[1]-eigenw[2]))/ (abs((eigenw[1]-eigenw[2]))+abs((eigenw[1]-eigenw[2]))+abs((eigenw[2]-eigenw[0]))))

        self._DC            =  M_DC1
        self._DC2           =  M_DC2
        self._DC3           =  M_DC3

        self._DC_percentage =  M_DC1_perc
        self._DC2_percentage=  M_DC2_perc

        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi

        M_iso_percentage     = int(M0_iso/M0 *100)
        self._iso_percentage = M_iso_percentage


        #self._seismic_moment   = N.sqrt(1./2*N.sum(eigenw**2) )
        self._seismic_moment   = M0
        self._moment_magnitude = N.log10(self._seismic_moment*1.0e7)/1.5 - 10.7

    #---------------------------------------------------------------

    def _M_to_principal_axis_system(self):
        """
        Read in Matrix M and set up eigenvalues (EW) and eigenvectors
        (EV) for setting up the principal axis system.

        The internal convention is the 'HNS'-system: H is the
        eigenvector for the smallest absolute eigenvalue, S is the
        eigenvector for the largest absolute eigenvalue, N is the null
        axis.

        Naming due to the geometry: a CLVD is
        Symmetric to the S-axis,
        Null-axis is common sense, and the third (auxiliary) axis
        Helps to construct the R³.

        Additionally builds matrix for basis transformation back to NED system.

        The eigensystem setup defines the colouring order for a later
        plotting in the BeachBall class. This order is set by the
        '_plot_clr_order' attribute.

        """

        M      = self._M
        M_devi = self._deviatoric



        # working in framework of 3 principal axes:
        # eigenvalues (EW) are in order from high to low
        # - neutral axis N, belongs to middle EW
        # - symmetry axis S ('sigma') belongs to EW with largest absolute value (P- or T-axis)
        # - auxiliary axis H ('help') belongs to remaining EW (T- or P-axis)
        
        #EW sorting from lowest to highest value
        EW_devi, EV_devi = N.linalg.eigh( M_devi )
        EW_order = N.argsort(EW_devi)

        #print 'order',EW_order

        if 1:#self._plot_isotropic_part:
            trace_M = N.trace(M)
            if abs(trace_M) < epsilon:
                trace_M = 0
            EW, EV = N.linalg.eigh( M )
            for i,ew in enumerate(EW):
                if abs(EW[i]) < epsilon:
                    EW[i] = 0
        else:
            trace_M = N.trace(M_devi)
            if abs(trace_M) < epsilon:
                trace_M = 0

            EW, EV = N.linalg.eigh( M_devi )
            for i,ew in enumerate(EW):
                if abs(EW[i]) < epsilon:
                    EW[i] = 0
        trace_M_devi =   N.trace(M_devi)


        EW1_devi = EW_devi[EW_order[0]]
        EW2_devi = EW_devi[EW_order[1]]
        EW3_devi = EW_devi[EW_order[2]]
        EV1_devi = EV_devi[:,EW_order[0]]
        EV2_devi = EV_devi[:,EW_order[1]]
        EV3_devi = EV_devi[:,EW_order[2]]


        EW1 = EW[EW_order[0]]
        EW2 = EW[EW_order[1]]
        EW3 = EW[EW_order[2]]
        EV1 = EV[:,EW_order[0]]
        EV2 = EV[:,EW_order[1]]
        EV3 = EV[:,EW_order[2]]

        chng_basis_tmp    =  N.matrix(N.zeros((3,3)))
        chng_basis_tmp[:,0] = EV1_devi
        chng_basis_tmp[:,1] = EV2_devi
        chng_basis_tmp[:,2] = EV3_devi
        det_mat =  N.linalg.det(chng_basis_tmp)

        symmetry_around_tension = 1
        clr = 1

        if abs(EW2_devi) < epsilon:
            EW2_devi = 0


        #implosion
        if EW1 < 0 and EW2 < 0 and EW3 < 0:
            symmetry_around_tension = 0
            #logger.debug( 'IMPLOSION - symmetry around pressure axis \n\n')
            clr = 1

        #explosion
        elif   EW1 > 0 and EW2 > 0 and EW3 > 0:
            symmetry_around_tension = 1
            if  abs(EW1_devi) > abs(EW3_devi):
                symmetry_around_tension = 0
            #logger.debug( 'EXPLOSION - symmetry around tension axis \n\n')
            clr = -1

        #net-implosion
        elif  EW2 < 0 and  sum([EW1,EW2,EW3]) < 0 :
            if  abs(EW1_devi) < abs(EW3_devi):
                symmetry_around_tension = 1
                clr = 1
            else:
                symmetry_around_tension = 1
                clr = 1

        #net-implosion
        elif  EW2_devi >= 0  and sum([EW1,EW2,EW3]) < 0 :
            symmetry_around_tension = 0
            clr = -1
            if  abs(EW1_devi) < abs(EW3_devi):
                symmetry_around_tension = 1
                clr = 1


        #net-explosion
        elif  EW2_devi < 0 and sum([EW1,EW2,EW3]) > 0 :
            symmetry_around_tension = 1
            clr = 1
            if  abs(EW1_devi) > abs(EW3_devi):
                symmetry_around_tension = 0
                clr = -1


        #net-explosion
        elif  EW2_devi >= 0 and sum([EW1,EW2,EW3]) > 0 :
            symmetry_around_tension = 0
            clr = -1


        else:
            #TODO check: this point should never be reached !!
            pass


        #test 26.9.2010:

        if abs(EW1_devi) < abs(EW3_devi):
            symmetry_around_tension = 1
            clr = 1
            if 0:#EW2 > 0 :#or (EW2 > 0 and EW2_devi > 0) :
                symmetry_around_tension = 0
                clr = -1

        if abs(EW1_devi) >= abs(EW3_devi):
            symmetry_around_tension = 0
            clr = -1
            if 0:#EW2 < 0 :
                symmetry_around_tension = 1
                clr = 1
        if (EW3 < 0 and N.trace(self._M) >= 0):
            #reaching this point means, we have a serious problem, likely of numerical nature
            print 'Houston, we have had a problem  - check M !!!!!! \n ( Trace(M) > 0, but largest eigenvalue is still negative)'
            raise MTError(' !! ')


        if trace_M == 0:
            #print 'pure deviatoric'
            if EW2 == 0:
                #print 'pure shear'
                symmetry_around_tension = 1
                clr = 1

            elif 2*abs(EW2) == abs(EW1) or 2*abs(EW2) == abs(EW3):
                #print 'pure clvd'
                if abs(EW1) < EW3:
                    #print 'CLVD: symmetry around tension'
                    symmetry_around_tension = 1
                    clr = 1
                else:
                    #print 'CLVD: symmetry around pressure'
                    symmetry_around_tension = 0
                    clr = -1
            else:
                #print 'mix of DC and CLVD'
                if abs(EW1) < EW3:
                    #print 'symmetry around tension'
                    symmetry_around_tension = 1
                    clr = 1
                else:
                    #print 'symmetry around pressure'
                    symmetry_around_tension = 0
                    clr = -1

        #define order of eigenvectors and values according to symmetry axis
        if symmetry_around_tension == 1:
            EWs = EW3.copy()
            EVs = EV3.copy()
            EWh = EW1.copy()
            EVh = EV1.copy()

        else:
            EWs = EW1.copy()
            EVs = EV1.copy()
            EWh = EW3.copy()
            EVh = EV3.copy()

        EWn = EW2
        EVn = EV2


        # build the basis system change matrix:
        chng_basis    =  N.matrix(N.zeros((3,3)))
        chng_fp_basis =  N.matrix(N.zeros((3,3)))


        #order of eigenvector's basis: (H,N,S)
        chng_basis[:,0] = EVh
        chng_basis[:,1] = EVn
        chng_basis[:,2] = EVs

        # matrix for basis transformation
        self._rotation_matrix = chng_basis

        #collections of eigenvectors and eigenvalues
        self._eigenvectors = [EVh,EVn,EVs]
        self._eigenvalues  = [EWh,EWn,EWs]

        #principal axes
        self._null_axis    = EVn
        self._t_axis       = EV1
        self._p_axis       = EV3

        #plotting order flag - important for plot in BeachBall class
        self._plot_clr_order = clr

        #print clr

        #collection of the faultplanes, given in strike, dip, slip-rake
        self._faultplanes = self._find_faultplanes()

    #---------------------------------------------------------------

    def _find_faultplanes(self):

        """
        Sets the two angle-triples, describing the faultplanes of the
        Double Couple, defined by the eigenvectors P and T of the
        moment tensor object.


        Define a reference Double Couple with strike = dip =
        slip-rake = 0, the moment tensor object's DC is transformed
        (rotated) w.r.t. this orientation. The respective rotation
        matrix yields the first fault plane angles as the Euler
        angles. After flipping the first reference plane by
        multiplying the appropriate flip-matrix, one gets the second fault
        plane's geometry.

        All output angles are in degree

        (
        to check:
        using Sebastian's conventions:

        rotationsmatrix1 = EV Matrix of M, but in order TNP (not as here  PNT!!!)

        reference-DC with strike, dip, rake = 0,0,0  in NED - form:  M = 0,0,0,0,-1,0

        the eigenvectors of this into a Matrix:

        trafo-matrix2 = EV Matrix of Reference-DC in order TNP

        effective Rotation matrix = (rotation_matrix1  * trafo-matrix2.T).T

        by checking for det <0, make sure, if  Matrix must be multiplied by -1 

        flip_matrix = 0,0,-1,0,-1,0,-1,0,0

        other DC orientation obtained by flip * effective Rotation matrix 

        both matrices in matrix_2_euler
        )

        """

        # reference Double Couple (in NED basis) - it has strike, dip, slip-rake = 0,0,0
        refDC                    = N.matrix( [[0.,0.,-1.],[0.,0.,0.],[-1.,0.,0.]], dtype=N.float )
        refDC_evals, refDC_evecs = N.linalg.eigh(refDC)

        #matrix which is turning from one fault plane to the other
        flip_dc                  = N.matrix( [[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]], dtype=N.float )

        #euler-tools need matrices of EV sorted in PNT:
        pnt_sorted_EV_matrix      = self._rotation_matrix.copy()

        #re-sort only necessary, if abs(p) <= abs(t)
        if self._plot_clr_order < 0:
            pnt_sorted_EV_matrix[:,0] = self._rotation_matrix[:,2]
            pnt_sorted_EV_matrix[:,2] = self._rotation_matrix[:,0]

        # rotation matrix, describing the rotation of the eigenvector
        # system of the input moment tensor into the eigenvector
        # system of the reference Double Couple
        rot_matrix_fp1       = (N.dot(pnt_sorted_EV_matrix, refDC_evecs.T)).T

        #check, if rotation has correct orientation
        if N.linalg.det(rot_matrix_fp1) < 0.:
            rot_matrix_fp1       *= -1.

        #adding a rotation into the (ambiguous) system of the second fault plane
        rot_matrix_fp2       = N.dot(flip_dc,rot_matrix_fp1)

        fp1                  = self._find_strike_dip_rake(rot_matrix_fp1)
        fp2                  = self._find_strike_dip_rake(rot_matrix_fp2)

        return  [fp1,fp2]

    #---------------------------------------------------------------
    def _find_strike_dip_rake(self,rotation_matrix):

        """
        Returns tuple of angles (strike, dip, slip-rake) in degrees, describing the fault plane.

        """

        (alpha, beta, gamma) = self._matrix_to_euler(rotation_matrix)


        return (beta*rad2deg, alpha*rad2deg, -gamma*rad2deg)

    #-----------------------

    def _cvec(self,x,y,z):
        """
        Builds a column vector (matrix type) from a 3 tuple.
        """
        return N.matrix( [[x,y,z]], dtype=N.float ).T

    #---------------------------------------------------------------


    def _matrix_to_euler(self, rotmat ):
        '''
        Returns three Euler angles alpha, beta, gamma (in radians) from a rotation matrix.

        '''

        ex = self._cvec(1.,0.,0.)
        ez = self._cvec(0.,0.,1.)
        exs = rotmat.T * ex
        ezs = rotmat.T * ez
        enodes = N.cross(ez.T,ezs.T).T
        if N.linalg.norm(enodes) < 1e-10:
            enodes = exs
        enodess = rotmat*enodes
        cos_alpha = float((ez.T*ezs))
        if cos_alpha > 1.: cos_alpha = 1.
        if cos_alpha < -1.: cos_alpha = -1.
        alpha = N.arccos(cos_alpha)
        beta  = N.mod( N.arctan2( enodes[1,0], enodes[0,0] ), N.pi*2. )
        gamma = N.mod( -N.arctan2( enodess[1,0], enodess[0,0] ), N.pi*2. )

        return self._unique_euler(alpha,beta,gamma)

    #---------------------------------------------------------------

    def _unique_euler(self, alpha, beta, gamma ):

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
    #---------------------------------------------------------------


    def _matrix_w_style_and_system(self, M2return, system, style='n'):
        """
        Returns the provided matrix transformed into the given basis system 'system'.

        If the argument 'style' is set to 'fancy', a 'print' of the return
        value yields a nice shell output of the matrix for better
        visual control.

        """

        M2return = basis_transform_matrix(M2return, 'NED', system.upper())

        if style.lower() in ['f','fan','fancy', 'y']:
            return fancy_matrix(M2return)
        else:
            return M2return


    #---------------------------------------------------------------

    def _vector_w_style_and_system(self, vectors, system, style='n'):
        """
        Returns the provided vector(s) transformed into the given basis system 'system'.

        If the argument 'style' is set to 'fancy', a 'print' of the return
        value yields a nice shell output of the vector(s) for better
        visual control.

        'vectors' can be either a single array, tuple, matrix or a collection in form of a list, array or matrix.
        If it's a list, each entry will be checked, if it's 3D - if not, an exception is raised.
        If it's a matrix or array with column-length 3, the columns are interpreted as vectors, otherwise, its transposed is used.

        """

        fancy = style.lower() in ['f','fan','fancy', 'y']

        lo_vectors = []

        # if list of vectors
        if type(vectors) == list:
            lo_vectors = vectors

        else:
            assert 3 in vectors.shape

            if N.shape(vectors)[0] == 3:
                for ii in  N.arange(N.shape(vectors)[1]) :
                    lo_vectors.append(vectors[:,ii])
            else:
                for ii in  N.arange(N.shape(vectors)[0]) :
                    lo_vectors.append(vectors[:,ii].transpose())

        lo_vecs_to_show = []
        for vec in lo_vectors:

            t_vec = basis_transform_vector(vec, 'NED', system.upper())

            if fancy:
                lo_vecs_to_show.append(fancy_vector(t_vec))
            else:
                lo_vecs_to_show.append(t_vec)


        if len(lo_vecs_to_show) == 1 :
            return lo_vecs_to_show[0]

        else:
            if fancy:
                return ''.join(lo_vecs_to_show)
            else:
                return lo_vecs_to_show

    #---------------------------------------------------------------

    def get_M(self,system='NED', style='n'):
        """
        Returns the moment tensor in matrix representation.

        Call with argument 'system' to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """

        if style== 'y':
            print '\n Full moment tensor in %s-coordinates: ' %(system)
            return  self._matrix_w_style_and_system(self._M,system,style)
        else:
            return  self._matrix_w_style_and_system(self._M,system,style)

    #---------------------------------------------------------------

    def get_decomposition(self,in_system='NED',out_system='NED', style='n'):
        """
        Returns a tuple of the decomposition results.

        Order:
        - 1 - basis of the provided input     (string)
        - 2 - basis of  the representation    (string)
        - 3 - chosen decomposition type      (integer)

        - 4 - full moment tensor              (matrix)

        - 5 - isotropic part                  (matrix)
        - 6 - isotropic percentage             (float)
        - 7 - deviatoric part                 (matrix)
        - 8 - deviatoric percentage            (float)

        - 9 - DC part                         (matrix)
        -10 - DC percentage                    (float)
        -11 - DC2 part                        (matrix)
        -12 - DC2 percentage                   (float)
        -13 - DC3 part                        (matrix)
        -14 - DC3 percentage                   (float)

        -15 - CLVD part                       (matrix)
        -16 - CLVD percentage                 (matrix)

        -17 - seismic moment                   (float)
        -18 - moment magnitude                 (float)

        -19 - eigenvectors                   (3-array)
        -20 - eigenvalues                       (list)
        -21 - p-axis                         (3-array)
        -22 - neutral axis                   (3-array)
        -23 - t-axis                         (3-array)


        """

        return  [in_system,out_system,self.get_decomp_type(),\
                 self.get_M(system=out_system),\
                 self.get_iso(system=out_system), self.get_iso_percentage(), \
                 self.get_devi(system=out_system), self.get_devi_percentage() ,\
                 self.get_DC(system=out_system), self.get_DC_percentage(), \
                 self.get_DC2(system=out_system), self.get_DC2_percentage(), \
                 self.get_DC3(system=out_system), self.get_DC3_percentage(), \
                 self.get_CLVD(system=out_system),self.get_CLVD_percentage(), \
                 self.get_moment(), self.get_mag(),\
                 self.get_eigvecs(system=out_system),self.get_eigvals(system=out_system),\
                 self.get_p_axis(system=out_system), self.get_null_axis(system=out_system), self.get_t_axis(system=out_system),\
                 self.get_fps()]

    #---------------------------------------------------------------

    def __str__(self):
        """
        Nice compilation of decomposition result to be viewed in the shell (call with 'print').
        """



        mexp = pow(10,N.ceil(N.log10(N.max(N.abs(self._M)))))

        m =  basis_transform_matrix(self._M/mexp, 'NED', self._output_basis)

        def b(i,j):
            x = self._output_basis.lower()
            return x[i]+x[j]

        s =  '\nScalar Moment: M0 = %g Nm (Mw = %3.1f)\n'
        s += 'Moment Tensor: M%s = %6.3f,  M%s = %6.3f, M%s = %6.3f,\n'
        s += '               M%s = %6.3f,  M%s = %6.3f, M%s = %6.3f    [ x %g ]\n\n'
        s = s % (self._seismic_moment, self._moment_magnitude,
            b(0,0), m[0,0],
            b(1,1), m[1,1],
            b(2,2), m[2,2],
            b(0,1), m[0,1],
            b(0,2), m[0,2],
            b(1,2), m[1,2],
            mexp)

        s += self._fault_planes_as_str()
        return s
    #---------------------------------------------------------------

    def _fault_planes_as_str(self):
        """
        Internal setup of a nice string, containing information about the fault planes.
        """
        s = '\n'
        for i,sdr in enumerate(self.get_fps()):
            s += 'Fault plane %i: strike = %3.0f°, dip = %3.0f°, slip-rake = %4.0f°\n' % \
                 (i+1, sdr[0], sdr[1], sdr[2])
        return s

   #---------------------------------------------------------------
    def get_input_system(self, style='n',**kwargs):
        """
        Returns the basis system of the input.
        """
        if style == 'y':
            print '\n Basis system of the input:\n   '
        return  self._input_basis
    #---------------------------------------------------------------
    def get_output_system(self, style='n',**kwargs):
        """
        Returns the basis system of the output.
        """
        if style == 'y':
            print '\n Basis system of the output: \n  '
        return  self._output_basis
    #---------------------------------------------------------------
    def get_decomp_type(self, style='n',**kwargs):
        """
        Returns the decomposition type.
        """

        if style == 'y':
            print '\n Decomposition type: \n  '
            return  MomentTensor.decomp_dict[self._decomposition_key][0]

        return  self._decomposition_key

    #---------------------------------------------------------------

    def get_iso(self,system='NED', style='n'):
        """
        Returns the isotropic part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Isotropic part in %s-coordinates: \n' %(system)
        return  self._matrix_w_style_and_system(self._isotropic,system,style)

    #---------------------------------------------------------------

    def get_devi(self,system='NED', style='n'):
        """
        Returns the  deviatoric part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Deviatoric part in %s-coordinates: \n' %(system)
        return  self._matrix_w_style_and_system(self._deviatoric,system,style)

    #---------------------------------------------------------------
    def get_DC(self,system='NED', style='n'):
        """
        Returns the  Double Couple  part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Double Couple part in %s-coordinates: \n ' %(system)

        return  self._matrix_w_style_and_system( self._DC,system,style)
    #---------------------------------------------------------------
    def get_DC2(self,system='NED', style='n'):
        """
        Returns the  second Double Couple  part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Second Double Couple part in %s-coordinates: \n' %(system)
        if  self._DC2==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''

        return  self._matrix_w_style_and_system( self._DC2,system,style)
    #---------------------------------------------------------------
    def get_DC3(self,system='NED', style='n'):
        """
        Returns the third Double Couple  part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Third Double Couple part in %s-coordinates: \n' %(system)

        if  self._DC3==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''
        return  self._matrix_w_style_and_system( self._DC3,system,style)

    #---------------------------------------------------------------
    def get_CLVD(self,system='NED', style='n'):
        """
        Returns the CLVD  part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n CLVD part in %s-coordinates: \n' %(system)
        if self._CLVD ==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''

        return  self._matrix_w_style_and_system( self._CLVD,system,style)

    #---------------------------------------------------------------
    def get_DC_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the DC part of the moment tensor in matrix representation.
        """

        if style == 'y':
            print '\n Double Couple percentage: \n'
        return self._DC_percentage
    #---------------------------------------------------------------
    def get_CLVD_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the DC part of the moment tensor in matrix representation.
        """

        if style == 'y':
            print '\n CLVD percentage: \n'
        if self._CLVD ==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''
        return int(100 - self._iso_percentage - self._DC_percentage)
    #---------------------------------------------------------------
    def get_DC2_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the second DC part of the moment tensor in matrix representation.
        """

        if style == 'y':
            print "\n Second Double Couple's percentage: \n"
        if  self._DC2==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''
        return self._DC2_percentage
    #---------------------------------------------------------------
    def get_DC3_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the third DC part of the moment tensor in matrix representation.
        """

        if style == 'y':
            print "\n Third Double Couple's percentage: \n"
        if  self._DC3==None:
            if style == 'y':
                print ' not available in this decomposition type '
            return ''
        return int(100 - self._DC2_percentage - self._DC_percentage)

    #---------------------------------------------------------------
    def get_iso_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the isotropic part of the moment tensor in matrix representation.
        """
        if style == 'y':
            print '\n Isotropic percentage: \n'
        return self._iso_percentage
    #---------------------------------------------------------------
    def get_devi_percentage(self,system='NED', style='n'):
        """
        Returns the percentage of the deviatoric part of the moment tensor in matrix representation.
        """
        if style == 'y':
            print '\n Deviatoric percentage: \n'
        return int(100-self._iso_percentage)

    #---------------------------------------------------------------
    def get_moment(self,system='NED', style='n'):
        """
        Returns the seismic moment (in Nm) of the moment tensor.
        """
        if style == 'y':
            print '\n Seismic moment (in Nm) : \n '
        return self._seismic_moment

    #---------------------------------------------------------------
    def get_mag(self,system='NED', style='n'):
        """
        Returns the  moment magnitude M_w of the moment tensor.
        """
        if style == 'y':
            print '\n Moment magnitude Mw: \n '
        return self._moment_magnitude

    #---------------------------------------------------------------
    def get_decomposition_key(self,system='NED', style='n'):
        """
        10 = standard decomposition (Jost & Herrmann)
        """
        if style == 'y':
            print '\n Decomposition key (standard = 10): \n '
        return self._decomposition_key

    #---------------------------------------------------------------

    def get_eigvals(self,system='NED',style='n',**kwargs):
        """
        Returns a list of the eigenvalues of the moment tensor.
        """
        if style=='y':
            if self._plot_clr_order < 0:
                print '\n Eigenvalues T N P :\n'

            else:
                print '\n Eigenvalues P N T :\n'

            return '%g, %g, %g' % tuple(self._eigenvalues)

        # in the order HNS:
        return self._eigenvalues

    #---------------------------------------------------------------
    def get_eigvecs(self,system='NED', style='n'):
        """
        Returns the eigenvectors  of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style=='y':

            if self._plot_clr_order < 0:
                print '\n Eigenvectors T N P (in basis system %s): '%(system)
            else:
                print '\n Eigenvectors P N T (in basis system %s): '%(system)

        return self._vector_w_style_and_system(self._eigenvectors,  system,style)

    #---------------------------------------------------------------
    def get_null_axis(self,system='NED', style='n'):
        """
        Returns the neutral axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """

        if style == 'y':
            print '\n Null-axis in %s -coordinates: '%(system)

        return self._vector_w_style_and_system(self._null_axis,  system,style)

    #---------------------------------------------------------------
    def get_t_axis(self,system='NED', style='n'):
        """
        Returns the tension axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n Tension-axis in %s -coordinates: '%(system)
        return self._vector_w_style_and_system(self._t_axis,  system,style)

    #---------------------------------------------------------------
    def get_p_axis(self,system='NED', style='n'):
        """
        Returns the pressure axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """

        if style == 'y':
            print '\n Pressure-axis in %s -coordinates: '%(system)
        return self._vector_w_style_and_system(self._p_axis,  system,style)

    #---------------------------------------------------------------
    def get_transform_matrix(self,system='NED', style='n'):
        """
        Returns the  transformation matrix (input system to principal axis system.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        if style == 'y':
            print '\n rotation matrix in %s -coordinates: '%(system)
        return  self._matrix_w_style_and_system(self._rotation_matrix,system,style)

    #---------------------------------------------------------------
    def get_fps(self,**kwargs):
        """
        Returns a list of the two faultplane 3-tuples, each showing strike, dip, slip-rake.
        """
        fancy_key = kwargs.get('style','0')
        if fancy_key[0].lower() == 'y':
            return self._fault_planes_as_str()
        else:
            return self._faultplanes
    #---------------------------------------------------------------
    def get_colour_order(self,**kwargs):
        """
        Returns the value of the plotting order (only important in BeachBall instances).
        """
        style = kwargs.get('style','0')[0].lower()
        if style == 'y':
            print '\n Colour order key: '
        return self._plot_clr_order


MomentTensor.decomp_dict = {
    1: ('ISO + DC + CLVD',                 MomentTensor._standard_decomposition),
    2: ('ISO + major DC + minor DC',       MomentTensor._decomposition_w_2DC),
    3: ('ISO + DC1 + DC2 + DC3',           MomentTensor._decomposition_w_3DC),
    4: ('ISO + strike DC + dip DC + CLVD', MomentTensor._decomposition_w_CLVD_2DC),
}


#---------------------------------------------------------------
#---------------------------------------------------------------
#
#   external functions:
#
#---------------------------------------------------------------


def strikediprake_2_moments(strike,dip,rake):
    """
    Return 6-tuple containing entries of M, calculated from fault plane angles (defined as in Jost&Herman), given in degrees.

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

    #print m
    #print norm_factor

    try:
        if  (norm_factor < 0.1) or ( norm_factor >= 10):
            if not abs(norm_factor) == 0:
                m = m/norm_factor

                return "\n  / %5.2F %5.2F %5.2F \\\n" % (m[0,0], m[0,1], m[0,2]) +\
                       "  | %5.2F %5.2F %5.2F  |   x  %F\n"  % (m[1,0], m[1,1], m[1,2], norm_factor) +\
                       "  \\ %5.2F %5.2F %5.2F /\n" % (m[2,0] ,m[2,1], m[2,2])
    except:
        pass



    return "\n  / %5.2F %5.2F %5.2F \\\n" % (m[0,0], m[0,1], m[0,2]) +\
           "  | %5.2F %5.2F %5.2F  | \n"  % (m[1,0], m[1,1], m[1,2]) +\
           "  \\ %5.2F %5.2F %5.2F /\n" % (m[2,0] ,m[2,1], m[2,2])


#-------------------------------------------------------------------

def fancy_vector(v):
    """

    Returns a given 3-vector or array in a cute way on the shell, if you use 'print' on the return value.

    """
    return "\n  / %5.2F \\\n" % (v[0]) +\
    "  | %5.2F  |\n"  % (v[1]) +\
    "  \\ %5.2F /\n" % (v[2])



####################################################################################################################
####################################################################################################################
#---------------------------------------------------------------
#---------------------------------------------------------------
#
#   Class for plotting:
#
#---------------------------------------------------------------



class BeachBall:
    """
    Class for generating a beachball projection for a provided moment tensor object.


    Input for instance generation: MomentTensor object [,keywords dictionary]

    Output can be plots of
    - the eigensystem
    - the complete sphere
    - the projection to a unit sphere
      ... either lower (standard) or upper half

    Beside the plots, the unit sphere projection may be saved in a given file.

    Alternatively, only the file can be provided without showing anything directly.

    """

    def __init__(self,MT=MomentTensor,kwargs_dict={}):

        self.MT = MT

        self._M = MT._M

        self._set_standard_attributes()

        self._update_attributes(kwargs_dict)

        self._nodallines_in_NED_system()


    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def ploBB(self,kwargs={}):
        """
        Method for plotting the projection of the beachball onto a unit sphere.

        Module matplotlib (pylab) must be installed !!!


        Input:
        dictionary with keywords
        

        """

        # updating keywords
        self._update_attributes(kwargs)

        # setting up the beachball geometry
        self._setup_BB()

        # generate plot of a unit sphere
        # either full beachball, or most likely projection of one hemisphere
        self._plot_US()

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def save_BB(self,kwargs={}):
        """
        Method for saving the 2D projection of the beachball without showing the plot.

        Module matplotlib (pylab) must be installed !!!

        Input:
        keyword dictionary

        required keyword arguments:

        - outfile : name of outfile, addressing w.r.t. current directory

        """

        # updating keywords
        self._update_attributes(kwargs)

        # setting up the beachball geometry
        self._setup_BB()

        # generate plot of a unit sphere and strore into file directly
        # either full beachball, or most likely projection of one hemisphere
        self._just_save_bb()

    #-------------------------------------------------------------------

    def _just_save_bb(self):
        """
        Internal method for saving the beachball unit sphere plot into a given  file.

        This method tries to setup the approprite backend according to the requested file format first. 'AGG' is used in most cases.
        """

        try:
            del matplotlib
        except:
            pass
        try:
            del pylab
        except:
            pass
        try:
            del P
        except:
            pass

        import matplotlib

        if self._plot_outfile_format == 'svg':
            try:
                 matplotlib.use('SVG')
            except:
                matplotlib.use('Agg')

        if self._plot_outfile_format == 'pdf':
            try:
                matplotlib.use('PDF')
            except:
                matplotlib.use('Agg')
                pass

        if self._plot_outfile_format == 'ps':
            try:
                matplotlib.use('PS')

            except:
                matplotlib.use('Agg')
                pass

        if self._plot_outfile_format == 'eps':
            try:
                matplotlib.use('Agg')

            except:
                matplotlib.use('PS')
                pass

        if self._plot_outfile_format == 'png':
            try:
                matplotlib.use('AGG')

            except:
                mp_out = matplotlib.use('GTKCairo')
                if mp_out:
                    mp_out2 = matplotlib.use('Cairo')
                    if mp_out2:
                        matplotlib.use('GDK')


        # finally generating the actual plot
        import pylab as P
        import os
        
        plotfig = self._setup_plot_US(P)

        outfile_format = self._plot_outfile_format
        outfile_name   = self._plot_outfile
        

        outfile_abs_name = os.path.realpath(os.path.abspath(os.path.join(os.curdir,outfile_name)))

        #save plot into file
        try:
            plotfig.savefig(outfile_abs_name, dpi=self._plot_dpi, transparent=True, format=outfile_format)

        except:
            print 'ERROR!! -- Saving of plot not possible'
            return

        #closing all opened plot windows
        P.close('all')

        #remove plot modules from memory
        del P
        del matplotlib

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def get_psxy(self,kwargs={}):
        """
        Method returning one single string, which can be piped into the psxy method of the GMT package.


        keyword arguments and defaults:


        - GMT_type             = fill/lines/EVs (select type of string - default = fill)
        - GMT_scaling          = 1.             (scale the beachball - original radius is 1)
        - GMT_tension_colour   = 1              (tension area of BB -- colour flag for -Z in psxy)
        - GMT_pressure_colour  = 0              (pressure area of BB -- colour flag for -Z in psxy)
        - GMT_show_2FPs        = 0              (flag, if both faultplanes are to be shown)
        - GMT_show_1FP         = 1              (flag, if one faultplane is to be shown)
        - GMT_FP_index         = 2              (which faultplane -- 1 or 2 )

        """

        self._GMT_type          = 'fill'
        self._GMT_2fps          = False
        self._GMT_1fp           = 0

        self._GMT_psxy_fill     = None
        self._GMT_psxy_nodals   = None
        self._GMT_psxy_EVs      = None
        self._GMT_scaling       = 1.

        self._GMT_tension_colour   = 1
        self._GMT_pressure_colour = 0


        self._update_attributes(kwargs)

        self._setup_BB()

        self._set_GMT_attributes()

        if self._GMT_type == 'fill':
            self._GMT_psxy_fill.seek(0)
            GMT_string  = self._GMT_psxy_fill.getvalue()
        elif self._GMT_type == 'lines':
            self._GMT_psxy_nodals.seek(0)
            GMT_string = self._GMT_psxy_nodals.getvalue()
        else:
            GMT_string = self._GMT_psxy_EVs.getvalue()

        return GMT_string

    #---------------------------------------------------------------

    def _add_2_GMT_string(self,FH_string,curve, colour):
        """
        Writes coordinate pair list of given curve  as string into temporal file handler.
        """

        colour_Z = colour

        wstring = '> -Z%i\n'%(colour_Z)
        FH_string.write(wstring)
        N.savetxt(FH_string, self._GMT_scaling*curve.transpose())

    #-------------------------------------------------------------------

    def  _set_GMT_attributes(self):
        """
        Set the beachball lines and nodals as strings into a file handler.
        """

        neg_nodalline = self._nodalline_negative_final_US
        pos_nodalline = self._nodalline_positive_final_US
        FP1_2_plot     = self._FP1_final_US
        FP2_2_plot     = self._FP2_final_US
        EV_2_plot      = self._all_EV_2D_US[:,:2].transpose()
        US             = self._unit_sphere

        tension_colour    = self._GMT_tension_colour
        pressure_colour  = self._GMT_pressure_colour

        #build strings for possible GMT-output, used by 'psxy'
        GMT_string_FH     = StringIO()
        GMT_linestring_FH = StringIO()
        GMT_EVs_FH        = StringIO()

        self._add_2_GMT_string(GMT_EVs_FH,EV_2_plot,tension_colour)
        GMT_EVs_FH.flush()


        if self._plot_clr_order > 0 :
            self._add_2_GMT_string(GMT_string_FH,US,pressure_colour)
            self._add_2_GMT_string(GMT_string_FH,neg_nodalline,tension_colour)
            self._add_2_GMT_string(GMT_string_FH,pos_nodalline,tension_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH,US,tension_colour)

                if self._plot_curve_in_curve < 1 :
                    self._add_2_GMT_string(GMT_string_FH,neg_nodalline,pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH,pos_nodalline,tension_colour)

                    GMT_string_FH.flush()

                else:
                    self._add_2_GMT_string(GMT_string_FH,pos_nodalline,pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH,neg_nodalline,tension_colour)

                    GMT_string_FH.flush()

        else:
            self._add_2_GMT_string(GMT_string_FH,US,tension_colour)
            self._add_2_GMT_string(GMT_string_FH,neg_nodalline,pressure_colour)
            self._add_2_GMT_string(GMT_string_FH,pos_nodalline,pressure_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH,US,pressure_colour)

                if self._plot_curve_in_curve < 1 :
                    self._add_2_GMT_string(GMT_string_FH,neg_nodalline,tension_colour)
                    self._add_2_GMT_string(GMT_string_FH,pos_nodalline,pressure_colour)

                    GMT_string_FH.flush()

                else:
                    self._add_2_GMT_string(GMT_string_FH,pos_nodalline,tension_colour)
                    self._add_2_GMT_string(GMT_string_FH,neg_nodalline,pressure_colour)

                    GMT_string_FH.flush()

        # set all nodallines and faultplanes for plotting:
        #
        self._add_2_GMT_string(GMT_linestring_FH,neg_nodalline,tension_colour)
        self._add_2_GMT_string(GMT_linestring_FH,pos_nodalline,tension_colour)


        if self._GMT_2fps :
            self._add_2_GMT_string(GMT_linestring_FH,FP1_2_plot,tension_colour)
            self._add_2_GMT_string(GMT_linestring_FH,FP2_2_plot,tension_colour)

        elif self._GMT_1fp:
            if not int(self._GMT_1fp) in [1,2]:
                print 'no fault plane specified for being plotted...continue without fault plane(s)'
                pass
            else:
                if int(self._GMT_1fp) == 1:
                    self._add_2_GMT_string(GMT_linestring_FH,FP1_2_plot,tension_colour)
                else:
                    self._add_2_GMT_string(GMT_linestring_FH,FP2_2_plot,tension_colour)

        self._add_2_GMT_string(GMT_linestring_FH,US,tension_colour)

        GMT_linestring_FH.flush()

        setattr(self,'_GMT_psxy_nodals',GMT_linestring_FH)
        setattr(self,'_GMT_psxy_fill',GMT_string_FH)
        setattr(self,'_GMT_psxy_EVs',GMT_EVs_FH)

    #-------------------------------------------------------------------
    def get_MT(self):
        """
        Returns the original moment tensor object, handed over to the class at generating this instance.
        """
        return self.MT

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def full_sphere_plot(self,kwargs={}):
        """
        Method for plotting the full beachball, projected on a circle with a radius 2.

        Module matplotlib (pylab) must be installed !!!


        Input:
        keyword dictionary


        required keyword arguments:
        none

        """


        self._update_attributes(kwargs)

        self._setup_BB()

        self._aux_plot()

    #-------------------------------------------------------------------

    def _aux_plot(self):
        """
        Generates the final plot of the total sphere (according to the chosen 2D-projection.

        """
        import matplotlib
        from matplotlib import interactive
        import pylab as P

        P.close('all')
        plotfig = P.figure(665,figsize=(self._plot_aux_plot_size,self._plot_aux_plot_size) )

        plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax = plotfig.add_subplot(111, aspect='equal')
        ax.axison = False

        EV_2_plot        = getattr(self,'_all_EV'+'_final')
        BV_2_plot        = getattr(self,'_all_BV'+'_final').transpose()
        curve_pos_2_plot = getattr(self,'_nodalline_positive'+'_final')
        curve_neg_2_plot = getattr(self,'_nodalline_negative'+'_final')
        FP1_2_plot       = getattr(self,'_FP1'+'_final')
        FP2_2_plot       = getattr(self,'_FP2'+'_final')

        tension_colour      =  self._plot_tension_colour
        pressure_colour    =  self._plot_pressure_colour



        if self._plot_clr_order > 0 :
            if self._plot_fill_flag:

                ax.fill( self._outer_circle[0,:],self._outer_circle[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)


                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0,:],self._outer_circle[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                    if self._plot_curve_in_curve < 1:
                        ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)


                    else:
                        ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)

            if self._plot_show_princ_axes:

                ax.plot( [EV_2_plot[0,0]],[EV_2_plot[1,0]],'m^',ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,3]],[EV_2_plot[1,3]],'mv',ms=self._plot_princ_axes_symsize ,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,1]],[EV_2_plot[1,1]],'b^',ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,4]],[EV_2_plot[1,4]],'bv',ms=self._plot_princ_axes_symsize ,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,2]],[EV_2_plot[1,2]],'g^',ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,5]],[EV_2_plot[1,5]],'gv',ms=self._plot_princ_axes_symsize ,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)

        else:
            if self._plot_fill_flag:
                ax.fill( self._outer_circle[0,:],self._outer_circle[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0,:],self._outer_circle[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                    if self._plot_curve_in_curve < 0 :
                        ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass
                    else:
                        ax.fill( curve_pos_2_plot[0,:],curve_pos_2_plot[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_neg_2_plot[0,:],curve_neg_2_plot[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass

            if self._plot_show_princ_axes:

                ax.plot( [EV_2_plot[0,0]],[EV_2_plot[1,0]],'g^',ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,3]],[EV_2_plot[1,3]],'gv',ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,1]],[EV_2_plot[1,1]],'b^',ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,4]],[EV_2_plot[1,4]],'bv',ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,2]],[EV_2_plot[1,2]],'m^',ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,5]],[EV_2_plot[1,5]],'mv',ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)

        self._plot_nodalline_colour='y'

        ax.plot( curve_neg_2_plot[0,:] ,curve_neg_2_plot[1,:],'o',c=self._plot_nodalline_colour,lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha ,ms=3 )

        self._plot_nodalline_colour='b'

        ax.plot( curve_pos_2_plot[0,:] ,curve_pos_2_plot[1,:],'D',c=self._plot_nodalline_colour,lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha ,ms=3)

        if self._plot_show_1faultplane:
            if self._plot_show_FP_index == 1:
                ax.plot( FP1_2_plot[0,:],FP1_2_plot[1,:],'+',c=self._plot_faultplane_colour,lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha,ms=5)

            elif self._plot_show_FP_index == 2:
                ax.plot( FP2_2_plot[0,:],FP2_2_plot[1,:],'+',c=self._plot_faultplane_colour,lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha,ms=5)

        elif self._plot_show_faultplanes :
            ax.plot( FP1_2_plot[0,:],FP1_2_plot[1,:],'+',c=self._plot_faultplane_colour,lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha,ms=4)
            ax.plot( FP2_2_plot[0,:],FP2_2_plot[1,:],'+',c=self._plot_faultplane_colour,lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha,ms=4)

        else:
            pass

        #if isotropic part shall be displayed, fill the circle completely with the appropriate colour
        if self._pure_isotropic:
            if abs( N.trace( self._M )) > epsilon:
                if self._plot_clr_order < 0:
                    ax.fill(self._outer_circle[0,:], self._outer_circle[1,:],fc=tension_colour, alpha= 1,zorder=100 )
                else:
                    ax.fill( self._outer_circle[0,:],self._outer_circle[1,:],fc=pressure_colour, alpha= 1,zorder=100 )

        #plot NED basis vectors
        if self._plot_show_basis_axes:

            plot_size_in_points = self._plot_size * 2.54 * 72
            points_per_unit = plot_size_in_points/2.

            fontsize  = plot_size_in_points / 66.
            symsize   = plot_size_in_points / 77.

            direction_letters = list('NSEWDU')
            for idx,val in enumerate(BV_2_plot):
                x_coord = val[0]
                y_coord = val[1]
                np_letter = direction_letters[idx]

                rot_angle    = - N.arctan2(y_coord,x_coord) + pi/2.
                original_rho = N.sqrt(x_coord**2 + y_coord**2)

                marker_x  = ( original_rho - ( 3* symsize  / points_per_unit ) ) * N.sin( rot_angle )
                marker_y  = ( original_rho - ( 3* symsize  / points_per_unit ) ) * N.cos( rot_angle )
                annot_x   = ( original_rho - ( 8.5* fontsize / points_per_unit ) ) * N.sin( rot_angle )
                annot_y   = ( original_rho - ( 8.5* fontsize / points_per_unit ) ) * N.cos( rot_angle )

                ax.text(annot_x,annot_y,np_letter,horizontalalignment='center', size=fontsize,weight='bold', verticalalignment='center',\
                        bbox=dict(edgecolor='white',facecolor='white', alpha=1))

                if original_rho > epsilon:
                    ax.scatter([marker_x],[marker_y],marker=(3,0,rot_angle) ,s=symsize**2,c='k',facecolor='k',zorder=300)
                else:
                    ax.scatter([x_coord],[y_coord],marker=(4,1,rot_angle) ,s=symsize**2,c='k',facecolor='k',zorder=300)



        #plot both circle lines (radius 1 and 2)
        ax.plot(self._unit_sphere[0,:],self._unit_sphere[1,:] ,c=self._plot_outerline_colour, lw=self._plot_outerline_width,alpha= self._plot_outerline_alpha*self._plot_total_alpha)# ,ls=':')
        ax.plot(self._outer_circle[0,:],self._outer_circle[1,:],c=self._plot_outerline_colour, lw=self._plot_outerline_width,alpha= self._plot_outerline_alpha*self._plot_total_alpha)# ,ls=':')

        #dummy points for setting plot plot size more accurately
        ax.plot([0,2.1,0,-2.1],[2.1,0,-2.1,0],',',alpha=0.)

        ax.autoscale_view(tight=True, scalex=True, scaley=True)
        interactive(True)


        if self._plot_save_plot:
            try:

                plotfig.savefig(self._plot_outfile+'.'+self._plot_outfile_format, dpi=self._plot_dpi, transparent=True, format=self._plot_outfile_format)

            except:
                print 'saving of plot not possible'

        P.show()

        del P
        del matplotlib

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def pa_plot(self,kwargs={}):
        """
        Method for plotting the BB in the principal axes system.

        Module matplotlib (pylab) must be installed !!!


        Input:
        keyword dictionary


        required keyword arguments:
        none


        """
        import matplotlib
        from matplotlib import interactive
        import pylab as P

        self._update_attributes(kwargs)


        r_hor     = self._r_hor_for_pa_plot
        r_hor_FP  = self._r_hor_FP_for_pa_plot

        P.rc('grid', color='#316931', linewidth=0.5, linestyle='-.')
        P.rc('xtick', labelsize=12)
        P.rc('ytick', labelsize=10)

        width, height = P.rcParams['figure.figsize']
        size = min(width, height)

        fig = P.figure(34,figsize=(size, size))
        P.clf()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='#d5de9c')

        r_steps  = [0.000001]
        for i in (N.arange(4)+1)*0.2:
            r_steps.append(i)
        r_labels = ['S']
        for ii in N.arange(len(r_steps)):
            if (ii+1)%2==0:
                r_labels.append(str(r_steps[ii]))
            else:
                r_labels.append(' ')


        t_angles = N.arange(0.,360.,90)
        t_labels = [' N ',' H ',' - N',' - H']

        P.thetagrids( t_angles, labels=t_labels )

        ax.plot(self._phi_curve, r_hor   , color='r', lw=3)
        ax.plot(self._phi_curve, r_hor_FP, color='b', lw=1.5)
        ax.set_rmax(1.0)
        P.grid(True)

        P.rgrids((r_steps),labels=r_labels)

        ax.set_title("beachball in eigenvector system", fontsize=15)



        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile+'.'+self._plot_outfile_format, dpi=self._plot_dpi, transparent=True, format=self._plot_outfile_format)

            except:
                print 'saving of plot not possible'

        P.show()


        del P
        del matplotlib

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------


    def _set_standard_attributes(self):
        """
        Sets default values of mandatory arguments.

        """

        #PLOT
        #
        #plot basis system and view point:
        self._plot_basis        = 'NED'
        self._plot_projection   = 'stereo'
        self._plot_viewpoint    = [0.,0.,0.]
        self._plot_basis_change = None

        #flag, if upper hemisphere is seen instead
        self._plot_show_upper_hemis  = False

        #flag, if isotropic part shall be considered
        self._plot_isotropic_part    = False
        self._pure_isotropic         = False

        #number of minimum points per line and full circle (number/360 is minimum of points per degree at rounded lines)
        self._plot_n_points          = 360

        #nodal line of pressure and tension regimes:
        self._plot_nodalline_width  = 2
        self._plot_nodalline_colour = 'k'
        self._plot_nodalline_alpha  = 1.

        #outer circle line
        self._plot_outerline_width   = 2
        self._plot_outerline_colour  = 'k'
        self._plot_outerline_alpha   = 1.

        #faultplane(s)
        self._plot_faultplane_width  = 4
        self._plot_faultplane_colour = 'b'
        self._plot_faultplane_alpha  = 1.

        self._plot_show_faultplanes  = False
        self._plot_show_1faultplane  = False
        self._plot_show_FP_index     = 1

        #principal  axes:
        self._plot_show_princ_axes   = False
        self._plot_princ_axes_symsize= 10
        self._plot_princ_axes_lw     = 3
        self._plot_princ_axes_alpha  = 0.5

        #NED basis:
        self._plot_show_basis_axes   = False


        #filling of the area:
        self._plot_clr_order         = self.MT.get_colour_order()
        self._plot_curve_in_curve    = 0
        self._plot_fill_flag         = True
        self._plot_tension_colour    = 'r'
        self._plot_pressure_colour   = 'w'
        self._plot_fill_alpha        = 1.

        #general plot options
        self._plot_size         = 5
        self._plot_aux_plot_size= 5
        self._plot_dpi          = 200

        self._plot_total_alpha  = 1.

        #possibility to add external data (e.g. measured polariations)
        self._plot_external_data= False
        self._external_data     = None

        #if, howto, whereto save the plot
        self._plot_save_plot         = False
        self._plot_outfile           = './BB_plot_example'
        self._plot_outfile_format    = 'png'

    #---------------------------------------------------------------

    def _update_attributes(self,kwargs):
        """
        Makes an internal update of the object's attributes with the
        provided list of keyword arguments.

        If the keyword (extended by a leading _ ) is in the dict of
        the object, the value is updated. Otherwise, the keyword is
        ignored.
        """
        for key in kwargs.keys():
            if key[0]=='_':
                kw = key[1:]
            else:
                kw=key
            if '_'+kw in dir(self) :
                setattr(self,'_'+kw, kwargs[key])

    #---------------------------------------------------------------

    def _setup_BB(self):
        """
        Setup of the beachball, when a plotting method is evoked.


        Contains all the technical stuff for generating the final view of the beachball:

        - Finding a rotation matrix, describing the given viewpoint onto the beachball projection
        - Rotating all elements (lines, points) w.r.t. the given viewpoint
        - Projecting the 3D sphere into the 2D plane
        - Building circle lines in radius r=1 and r=2
        - Correct the order of line points, yielding a consecutive set of points for drawing lines
        - Smoothing of all curves, avoiding nasty sectioning connection lines
        - Checking, if the two nodalline curves are laying completely within each other ( cahnges plotting order of overlay plot construction)
        - Projection of final smooth solution onto the standard unit sphere
        """



        self._find_basis_change_2_new_viewpoint()

        self._rotate_all_objects_2_new_view()

        self._vertical_2D_projection()

        self._build_circles()

        if not  self.MT._iso_percentage == 100:

            self._correct_curves()

            self._smooth_curves()

            self._check_curve_in_curve()

        self._projection_2_unit_sphere()

        if self.MT._iso_percentage == 100:
            if N.trace(self.MT.get_M()) < 0:
                self._plot_clr_order = 1
            else:
                self._plot_clr_order = -1


    #---------------------------------------------------------------


    def _correct_curves(self):
        """
        Correcting potentially wrong curves.

        Checks, if the order of the given coordinates of the lines must be re-arranged, allowing for an automatical line plotting.

        """

        list_of_curves_2_correct = ['nodalline_negative', 'nodalline_positive','FP1','FP2']
        projection = self._plot_projection

        n_curve_points = self._plot_n_points

        for obj in list_of_curves_2_correct:
            obj2cor_name = '_'+obj+'_2D'
            obj2cor = getattr(self,obj2cor_name)

            obj2cor_in_right_order = self._sort_curve_points(obj2cor)

            # check, if curve closed !!!!!!
            start_r               = N.sqrt(obj2cor_in_right_order[0,0]**2  + obj2cor_in_right_order[1,0]**2 )
            r_last_point          = N.sqrt(obj2cor_in_right_order[0,-1]**2 + obj2cor_in_right_order[1,-1]**2   )
            dist_last_first_point = N.sqrt( (obj2cor_in_right_order[0,-1] - obj2cor_in_right_order[0,0])**2 + (obj2cor_in_right_order[1,-1] - obj2cor_in_right_order[1,0] )**2 )


            # check, if distance between last and first point is smaller than the distance between last point and the edge (at radius=2)
            if dist_last_first_point > (2 - r_last_point):
                #add points on edge to polygon, if it is an open curve
                phi_end   = N.arctan2(obj2cor_in_right_order[0,-1],obj2cor_in_right_order[1,-1])% (2*pi)
                R_end     = r_last_point
                phi_start = N.arctan2(obj2cor_in_right_order[0,0],obj2cor_in_right_order[1,0])% (2*pi)
                R_start   = start_r


                #add one point on the edge every fraction of degree given by input parameter, increase the radius linearily
                phi_end_larger   = N.sign(phi_end-phi_start)
                angle_smaller_pi = N.sign( pi - N.abs(phi_end - phi_start) )

                if phi_end_larger * angle_smaller_pi > 0:
                    go_ccw = True
                    openangle =  (phi_end - phi_start)%(2*pi)
                else:
                    go_ccw = False
                    openangle =  (phi_start - phi_end)%(2*pi)

                radius_interval = R_start - R_end # closing from end to start

                n_edgepoints = int(openangle*rad2deg * n_curve_points/360.) - 1


                if go_ccw:
                    obj2cor_in_right_order = list(obj2cor_in_right_order.transpose())
                    for kk in N.arange(n_edgepoints)+1:
                        current_phi    = phi_end - kk*openangle/(n_edgepoints+1)
                        current_radius = R_end + kk*radius_interval/(n_edgepoints+1)
                        obj2cor_in_right_order.append([current_radius*N.sin(current_phi), current_radius*N.cos(current_phi) ]   )
                    obj2cor_in_right_order = N.array(obj2cor_in_right_order).transpose()
                else:
                    obj2cor_in_right_order = list(obj2cor_in_right_order.transpose())
                    for kk in N.arange(n_edgepoints)+1:
                        current_phi = phi_end + kk*openangle/(n_edgepoints+1)
                        current_radius = R_end + kk*radius_interval/(n_edgepoints+1)
                        obj2cor_in_right_order.append([current_radius*N.sin(current_phi), current_radius*N.cos(current_phi) ]   )
                    obj2cor_in_right_order = N.array(obj2cor_in_right_order).transpose()


            setattr(self,'_'+obj+'_in_order',obj2cor_in_right_order)

        return 1



    #---------------------------------------------------------------

    def _nodallines_in_NED_system(self):
        """
        The two nodal lines between the areas on a beachball are given by the points, where
        tan**2(alpha) = (-EWs/(EWN*cos(phi)**2 + EWh*sin(phi)**2))
        is fulfilled.

        This solution is gained in the principal axes system and then expressed in terms of the NED basis system

        output:
        - set of points, building the first nodal line,  coordinates in the input basis system (standard NED)
        - set of points, building the second nodal line,  coordinates in the input basis system (standard NED)
        - array with 6 points, describing positive and negative part of 3 principal axes
        - array with partition of full circle (angle values in degrees) fraction is given by parametre n_curve_points
        """

        # build the nodallines of positive/negative areas in the principal axes system

        n_curve_points = self._plot_n_points

        # phi is the angle between neutral axis and horizontal projection
        # of the curve point to the surface, spanned by H- and
        # N-axis. Running mathematically negative (clockwise) around the
        # SIGMA-axis. Stepsize is given by the parametre for number of
        # curve points
        phi   = (N.arange(n_curve_points)/float(n_curve_points) + 1./n_curve_points )*2*pi
        self._phi_curve           = phi

        # analytical/geometrical solution for separatrix curve - alpha is opening angle
        # between principal axis SIGMA and point of curve. (alpha is 0, if
        # curve lies directly on the SIGMA axis)

        # CASE: including isotropic part
        # sigma axis flippes, if EWn flippes sign

        #--------------------------------------------------------------------------------------------------
        EWh_devi = self.MT.get_eigvals()[0] - 1./3 * N.trace(self._M)
        EWn_devi = self.MT.get_eigvals()[1] - 1./3 * N.trace(self._M)
        EWs_devi = self.MT.get_eigvals()[2] - 1./3 * N.trace(self._M)

        if not self._plot_isotropic_part:
            EWh = EWh_devi
            EWn = EWn_devi
            EWs = EWs_devi


        else:

            EWh_tmp = self.MT.get_eigvals()[0]# - 1./3 * N.trace(self._M)
            EWn_tmp = self.MT.get_eigvals()[1]# - 1./3 * N.trace(self._M)
            EWs_tmp = self.MT.get_eigvals()[2]# - 1./3 * N.trace(self._M)

            trace_m = N.sum(self.MT.get_eigvals())
            EWh = EWh_tmp.copy()
            EWs = EWs_tmp.copy()

            if trace_m !=0 :
                if (self._plot_clr_order > 0 and EWn_tmp >=0 and abs(EWs_tmp)>abs(EWh_tmp)) or (self._plot_clr_order < 0 and EWn_tmp <=0 and abs(EWs_tmp)>abs(EWh_tmp)):
                    EWs = EWh_tmp.copy()
                    EWh = EWs_tmp.copy()
                    #print 'changed order!!\n'
                    EVs_tmp = self.MT._rotation_matrix[:,2].copy()
                    EVh_tmp = self.MT._rotation_matrix[:,0].copy()

                    self.MT._rotation_matrix[:,0]  = EVs_tmp
                    self.MT._rotation_matrix[:,2]  = EVh_tmp
                    self._plot_clr_order *= -1


            EWn = EWn_tmp.copy()


        if abs(EWn) < epsilon:
            EWn = 0
        norm_factor =  max(N.abs([EWh,EWn,EWs]))

        [EWh,EWn,EWs] = [xx /norm_factor  for xx in [EWh,EWn,EWs] ]


        RHS   = -EWs /(EWn * N.cos(phi)**2 + EWh * N.sin(phi)**2 )

        if N.all([N.sign(xx)>=0 for xx in RHS ]):
            alpha = N.arctan(N.sqrt(RHS) ) *rad2deg
        else:
            alpha = phi.copy()
            alpha[:] = 90
            self._pure_isotropic = 1

        #fault planes:
        RHS_FP   = 1. /( N.sin(phi)**2 )
        alpha_FP = N.arctan(N.sqrt(RHS_FP) ) *rad2deg


        # horizontal coordinates of curves
        r_hor             =  N.sin(alpha /rad2deg)
        r_hor_FP          =  N.sin(alpha_FP /rad2deg)

        self._r_hor_for_pa_plot    = r_hor
        self._r_hor_FP_for_pa_plot = r_hor_FP


        H_values          =  N.sin(phi) * r_hor
        N_values          =  N.cos(phi) * r_hor
        H_values_FP       =  N.sin(phi) * r_hor_FP
        N_values_FP       =  N.cos(phi) * r_hor_FP



        # set vertical value of curve point coordinates - two symmetric curves exist
        S_values_positive    =  N.cos(alpha/rad2deg)
        S_values_negative    = -N.cos(alpha/rad2deg)
        S_values_positive_FP =  N.cos(alpha_FP/rad2deg)
        S_values_negative_FP = -N.cos(alpha_FP/rad2deg)


        #############
        # change basis back to original input reference system
        #########

        chng_basis = self.MT._rotation_matrix

        line_tuple_pos = N.zeros((3,n_curve_points ))
        line_tuple_neg = N.zeros((3,n_curve_points ))


        for ii in N.arange(n_curve_points):
            pos_vec_in_EV_basis  = N.array([H_values[ii],N_values[ii],S_values_positive[ii] ]).transpose()
            neg_vec_in_EV_basis  = N.array([H_values[ii],N_values[ii],S_values_negative[ii] ]).transpose()
            line_tuple_pos[:,ii] = N.dot(chng_basis, pos_vec_in_EV_basis )
            line_tuple_neg[:,ii] = N.dot(chng_basis, neg_vec_in_EV_basis )

        EVh = self.MT.get_eigvecs()[0]
        EVn = self.MT.get_eigvecs()[1]
        EVs = self.MT.get_eigvecs()[2]

        all_EV = N.zeros((3,6))

        EVh_orig     = N.dot(chng_basis, EVs)
        all_EV[:,0]  =  EVh.transpose() #_orig.transpose()
        EVn_orig     = N.dot(chng_basis, EVn)
        all_EV[:,1]  = EVn.transpose() # _orig.transpose()
        EVs_orig     = N.dot(chng_basis, EVh)
        all_EV[:,2]  = EVs.transpose() # _orig.transpose()
        EVh_orig_neg = N.dot(chng_basis, EVs)
        all_EV[:,3]  = -EVh.transpose() #_orig_neg.transpose()
        EVn_orig_neg = N.dot(chng_basis, EVn)
        all_EV[:,4]  = -EVn.transpose() # _orig_neg.transpose()
        EVs_orig_neg = N.dot(chng_basis, EVh)
        all_EV[:,5]  = -EVs.transpose() # _orig_neg.transpose()


        #basis vectors:
        all_BV       = N.zeros((3,6))
        all_BV[:,0]  = N.array((1,0,0))
        all_BV[:,1]  = N.array((-1,0,0))
        all_BV[:,2]  = N.array((0,1,0))
        all_BV[:,3]  = N.array((0,-1,0))
        all_BV[:,4]  = N.array((0,0,1))
        all_BV[:,5]  = N.array((0,0,-1))


        #re-sort the two 90 degree nodal lines to 2 fault planes -> cut each at
        #halves and merge pairs
        #additionally change basis system to NED reference system

        midpoint_idx = int(n_curve_points/2.)

        FP1 = N.zeros((3,n_curve_points ))
        FP2 = N.zeros((3,n_curve_points ))

        for ii in N.arange(midpoint_idx):
            FP1_vec   = N.array([H_values_FP[ii],N_values_FP[ii],S_values_positive_FP[ii] ]).transpose()
            FP2_vec   = N.array([H_values_FP[ii],N_values_FP[ii],S_values_negative_FP[ii] ]).transpose()
            FP1[:,ii] = N.dot(chng_basis, FP1_vec )
            FP2[:,ii] = N.dot(chng_basis, FP2_vec )

        for jj in N.arange(midpoint_idx):
            ii = n_curve_points - jj - 1

            FP1_vec = N.array([H_values_FP[ii],N_values_FP[ii],S_values_negative_FP[ii] ]).transpose()
            FP2_vec = N.array([H_values_FP[ii],N_values_FP[ii],S_values_positive_FP[ii] ]).transpose()
            FP1[:,ii] = N.dot(chng_basis, FP1_vec )
            FP2[:,ii] = N.dot(chng_basis, FP2_vec )

        #identify with faultplane index, gotten from 'get_fps':
        self._FP1                 = FP1
        self._FP2                 = FP2

        self._all_EV              = all_EV
        self._all_BV              = all_BV
        self._nodalline_negative = line_tuple_neg
        self._nodalline_positive = line_tuple_pos


    #---------------------------------------------------------------
    def _identify_faultplanes(self):

        """
        See, if the 2 faultplanes, given as attribute of the moment
        tensor object, handed to this instance, are consistent with
        the faultplane lines, obtained from the basis solution. If
        not, interchange the indices of the newly found ones.

        """
        # TODO !!!!!!

        pass
    #---------------------------------------------------------------

    def _find_basis_change_2_new_viewpoint(self):
        """
        Finding the Eulerian angles, if you want to rotate an object.

        Your original view point is the position (0,0,0). Input are the
        coordinates of the new point of view, equivalent to geographical
        coordinates.


        Example:

        Original view onto the Earth is from right above lat=0, lon=0 with
        north=upper edge, south=lower edge. Now you want to see the Earth
        from a position somewhere near Baku. So lat=45,
        lon=45, azimuth=0.

        The Earth must be rotated around some axis, not to be determined.
        The rotation matrixx is the matrix for the change of basis to the
        new local orthonormal system.


        input:
        -- latitude in degrees from -90 (south) to 90 (north)
        -- longitude in degrees from -180 (west) to 180 (east)
        -- azimuth in degrees from 0 (heading north) to 360 (north again)
        """


        new_latitude  = self._plot_viewpoint[0]
        new_longitude = self._plot_viewpoint[1]
        new_azimuth   = self._plot_viewpoint[2]

        s_lat = N.sin(new_latitude/rad2deg)
        if abs(s_lat) < epsilon :
            s_lat = 0
        c_lat = N.cos(new_latitude/rad2deg)
        if abs(c_lat) < epsilon :
            c_lat = 0
        s_lon = N.sin(new_longitude/rad2deg)
        if abs(s_lon) < epsilon :
            s_lon = 0
        c_lon = N.cos(new_longitude/rad2deg)
        if abs(c_lon) < epsilon :
            c_lon = 0

        # assume input basis as NED!!!

        # original point of view therein is (0,0,-1)
        # new point at lat=latitude, lon=longitude, az=0, given in old NED-coordinates:
        # (cos(latitude), sin(latitude)*sin(longitude), sin(latitude)*cos(longitude) )
        #
        # new " down' " is given by the negative position vector, so pointing inwards to the centre point
        #down_prime = - ( N.array( ( s_lat, c_lat*c_lon, -c_lat*s_lon ) ) )
        down_prime = - ( N.array( ( s_lat, c_lat*s_lon , -c_lat*c_lon) ) )

        #normalise:
        down_prime /= N.sqrt(N.dot(down_prime,down_prime))

        #print down_prime
        # get second local basis vector " north' " by orthogonalising (Gram-Schmidt method) the original north w.r.t. the new " down' "
        north_prime_not_normalised     = N.array( (1.,0.,0.) ) - ( N.dot(down_prime,  N.array( (1.,0.,0.) ) )/(N.dot(down_prime,down_prime)) * down_prime)

        len_north_prime_not_normalised = N.sqrt(N.dot(north_prime_not_normalised,north_prime_not_normalised))
        # check for poles:
        if N.abs(len_north_prime_not_normalised) < epsilon:
            #case: north pole
            if s_lat > 0 :
                north_prime                    =  N.array( (0.,0.,1.) )
            #case: south pole
            else:
                north_prime                    =  N.array( (0.,0.,-1.) )
        else:
            north_prime                    = north_prime_not_normalised / len_north_prime_not_normalised

        # third basis vector is obtained by a cross product of the first two
        east_prime                 = N.cross(down_prime,north_prime)

        #normalise:
        east_prime    /=  N.sqrt(N.dot(east_prime,east_prime))

        rotmat_pos_raw      =  N.zeros((3,3))
        rotmat_pos_raw[:,0] =  north_prime
        rotmat_pos_raw[:,1] =  east_prime
        rotmat_pos_raw[:,2] =  down_prime

        rotmat_pos = N.matrix(rotmat_pos_raw).T
        # this matrix gives the coordinates of a given point in the old coordinates w.r.t. the new system

        #up to here, only the position has changed, the angle of view
        #(azimuth) has to be added by an additional rotation around the
        #down'-axis (in the frame of the new coordinates)

        # set up the local rotation around the new down'-axis by the given
        # angle 'azimuth'. Positive values turn view counterclockwise from the new
        # north'
        only_rotation = N.zeros((3,3))
        s_az = N.sin(new_azimuth/rad2deg)
        if abs(s_az) < epsilon:
            s_az = 0.
        c_az = N.cos(new_azimuth/rad2deg)
        if abs(c_az) < epsilon:
           c_az = 0.

        only_rotation[2,2] = 1
        only_rotation[0,0] = c_az
        only_rotation[1,1] = c_az
        only_rotation[0,1] = -s_az
        only_rotation[1,0] = s_az

        local_rotation = N.matrix(only_rotation)

        #apply rotation from left!!
        total_rotation_matrix = N.dot(local_rotation,rotmat_pos)


        # yields the complete matrix for representing the old coordinates in the new (rotated) frame:
        self._plot_basis_change = total_rotation_matrix



    #---------------------------------------------------------------
    def _rotate_all_objects_2_new_view(self):
        """
        Rotate all relevant parts of the solution - namely the
        eigenvector-projections, the 2 nodallines, and the faultplanes
        - so that they are seen from the new viewpoint.

        """
        objects_2_rotate = ['all_EV','all_BV','nodalline_negative', 'nodalline_positive','FP1','FP2']

        for obj in objects_2_rotate:

            object2rotate = getattr(self,'_'+obj).transpose()

            #logger.debug( str(N.shape(object2rotate)),str(len(object2rotate)) )
            #logger.debug( str(N.shape(self._plot_basis_change)) )

            rotated_thing = object2rotate.copy()
            for i in N.arange(len(object2rotate)):
                rotated_thing[i] = N.dot(self._plot_basis_change,object2rotate[i])

            rotated_object = rotated_thing.copy()
            setattr(self,'_'+obj+'_rotated',rotated_object.transpose())

    #---------------------------------------------------------------

    def _vertical_2D_projection(self):
        """
        Start the vertical projection of the 3D beachball onto the 2D plane.

        The projection is chosen according to the attribute '_plot_projection'
        """


        list_of_possible_projections = ['stereo','ortho','lambert','gnom']

        if not self._plot_projection in list_of_possible_projections:
            print 'requested projection not possible - choose from:\n ',list_of_possible_projections
            raise MTError(' !! ')

        if self._plot_projection == 'stereo':
            if not self._stereo_vertical():
                print 'ERROR in stereo_vertical'
                raise MTError(' !! ')

        if self._plot_projection == 'ortho':
            if not self._orthographic_vertical():
                print 'ERROR in stereo_vertical'
                raise MTError(' !! ')

        if self._plot_projection == 'lambert':
            if not self._lambert_vertical():
                print 'ERROR in stereo_vertical'
                raise MTError(' !! ')

        if self._plot_projection == 'gnom':
            if not self._gnomonic_vertical():
                print 'ERROR in stereo_vertical'
                raise MTError(' !! ')


    #---------------------------------------------------------------
    def _stereo_vertical(self):

        """
        Stereographic/azimuthal conformal 2D projection onto a plane, tangent to the lowest point (0,0,1).

        Keeps the angles constant!

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV','all_BV','nodalline_negative', 'nodalline_positive','FP1','FP2']

        available_coord_systems = ['NED']

        if not self._plot_basis in available_coord_systems:
            print 'requested plotting projection not possible - choose from :\n',avail_coord_systems
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_'+obj+'_rotated'
            o2proj   = getattr(self, obj_name)
            coords   = o2proj.copy()

            n_points = len(o2proj[0,:])
            stereo_coords = N.zeros((2,n_points))

            for ll in N.arange(n_points):
                # second component is EAST
                co_x = coords[1,ll]

                # first component is NORTH
                co_y = coords[0,ll]

                # z given in DOWN
                co_z = -coords[2,ll]


                rho_hor = N.sqrt(co_x**2 + co_y**2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if  co_z < 0:
                            new_x = 2
                    else:
                        if  co_z > 0:
                            new_x = 2
                else:
                    if co_z < 0:
                        new_rho =      rho_hor/(1.-co_z)
                        if plot_upper_hem:
                            new_rho = 2 - (rho_hor/(1.-co_z))

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho

                    else:
                        new_rho = 2 - (rho_hor/(1.+co_z))
                        if plot_upper_hem:
                            new_rho =      rho_hor/(1.+co_z)

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho


                stereo_coords[0,ll] = new_x
                stereo_coords[1,ll] = new_y


            setattr(self,'_'+obj+'_2D',stereo_coords)
            setattr(self,'_'+obj+'_final',stereo_coords)

        return 1

    #---------------------------------------------------------------
    def _orthographic_vertical(self):

        """
        Orthographic 2D projection onto a plane, tangent to the lowest point (0,0,1).

        Shows the natural view on a 2D sphere from large distances (assuming parallel projection)

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV','all_BV','nodalline_negative', 'nodalline_positive','FP1','FP2']

        available_coord_systems = ['NED']

        if not self._plot_basis in available_coord_systems:
            print 'requested plotting projection not possible - choose from :\n',avail_coord_systems
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_'+obj+'_rotated'
            o2proj   = getattr(self, obj_name)
            coords   = o2proj.copy()

            n_points = len(o2proj[0,:])
            coords2D = N.zeros((2,n_points))

            for ll in N.arange(n_points):
                # second component is EAST
                co_x = coords[1,ll]

                # first component is NORTH
                co_y = coords[0,ll]

                # z given in DOWN
                co_z = -coords[2,ll]


                rho_hor = N.sqrt(co_x**2 + co_y**2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if  co_z < 0:
                            new_x = 2
                    else:
                        if  co_z > 0:
                            new_x = 2


                else:
                    if co_z < 0:
                        new_rho =      rho_hor
                        if plot_upper_hem:
                            new_rho = 2 - rho_hor

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho

                    else:
                        new_rho = 2 - rho_hor
                        if plot_upper_hem:
                            new_rho =      rho_hor

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho


                coords2D[0,ll] = new_x
                coords2D[1,ll] = new_y


            setattr(self,'_'+obj+'_2D',coords2D)
            setattr(self,'_'+obj+'_final',coords2D)

        return 1

    #---------------------------------------------------------------

    def _lambert_vertical(self):

        """
        Lambert azimuthal equal-area 2D projection onto a plane, tangent to the lowest point (0,0,1).

        Keeps the area constant!

        The parts in the lower hemisphere are projected to the unit
        sphere (only here the area is kept constant), the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV','all_BV','nodalline_negative', 'nodalline_positive','FP1','FP2']

        available_coord_systems = ['NED']

        if not self._plot_basis in available_coord_systems:
            print 'requested plotting projection not possible - choose from :\n',avail_coord_systems
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_'+obj+'_rotated'
            o2proj   = getattr(self, obj_name)
            coords   = o2proj.copy()

            n_points = len(o2proj[0,:])
            coords2D = N.zeros((2,n_points))

            for ll in N.arange(n_points):
                # second component is EAST
                co_x = coords[1,ll]

                # first component is NORTH
                co_y = coords[0,ll]

                # z given in DOWN
                co_z = -coords[2,ll]

                rho_hor = N.sqrt(co_x**2 + co_y**2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if  co_z < 0:
                            new_x = 2
                    else:
                        if  co_z > 0:
                            new_x = 2



                else:
                    if co_z < 0:
                        new_rho =      rho_hor/N.sqrt(1.-co_z)

                        if plot_upper_hem:
                            new_rho = 2 - (rho_hor/N.sqrt(1.-co_z))

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho

                    else:
                        new_rho = 2 - (rho_hor/N.sqrt(1.+co_z))

                        if plot_upper_hem:
                            new_rho =      rho_hor/N.sqrt(1.+co_z)

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho


                coords2D[0,ll] = new_x
                coords2D[1,ll] = new_y


            setattr(self,'_'+obj+'_2D',coords2D)
            setattr(self,'_'+obj+'_final',coords2D)

        return 1


    #---------------------------------------------------------------
    def _gnomonic_vertical(self):

        """
        Gnomonic 2D projection onto a plane, tangent to the lowest point (0,0,1).

        Keeps the great circles as straight lines (geodetics constant) !

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV','all_BV','nodalline_negative', 'nodalline_positive','FP1','FP2']

        available_coord_systems = ['NED']

        if not self._plot_basis in available_coord_systems:
            print 'requested plotting projection not possible - choose from :\n',avail_coord_systems
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_'+obj+'_rotated'
            o2proj   = getattr(self, obj_name)
            coords   = o2proj.copy()

            n_points = len(o2proj[0,:])
            coords2D = N.zeros((2,n_points))


            for ll in N.arange(n_points):
                # second component is EAST
                co_x = coords[1,ll]

                # first component is NORTH
                co_y = coords[0,ll]

                # z given in DOWN
                co_z = -coords[2,ll]


                rho_hor = N.sqrt(co_x**2 + co_y**2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if  co_z > 0:
                        new_x = 2
                        if plot_upper_hem:
                            new_x = 0

                else:
                    if co_z < 0:
                        new_rho =     N.cos(N.arcsin(rho_hor)) *N.tan(N.arcsin(rho_hor))

                        if plot_upper_hem:
                            new_rho = 2 - (   N.cos(N.arcsin(rho_hor)) * N.tan(N.arcsin(rho_hor))  )

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho

                    else:
                        new_rho =  2 - ( N.cos(N.arcsin(rho_hor))* N.tan(N.arcsin(rho_hor)))

                        if plot_upper_hem:
                            new_rho =    N.cos(N.arcsin(rho_hor)) * N.tan(N.arcsin(rho_hor))

                        new_x   = co_x /rho_hor * new_rho
                        new_y   = co_y /rho_hor * new_rho


                coords2D[0,ll] = new_x
                coords2D[1,ll] = new_y


            setattr(self,'_'+obj+'_2D',coords2D)
            setattr(self,'_'+obj+'_final',coords2D)

        return 1

    #---------------------------------------------------------------

    def _build_circles(self):
        """
        Sets two sets of points, describing the unit sphere and the outer circle with r=2.

        Added as attributes '_unit_sphere' and '_outer_circle'.
        """


        phi = self._phi_curve

        UnitSphere      = N.zeros((2,len(phi)))
        UnitSphere[0,:] = N.cos(phi)
        UnitSphere[1,:] = N.sin(phi)

        # outer circle ( radius for stereographic projection is set to 2 )
        outer_circle_points      = 2 * UnitSphere

        self._unit_sphere  = UnitSphere
        self._outer_circle = outer_circle_points


    #---------------------------------------------------------------
    def _sort_curve_points(self,curve):
        """
        Checks, if curve points are in right order for line plotting.

        If not, a re-arranging is carried out.

        """


        sorted_curve = N.zeros((2,len(curve[0,:])))

        #in polar coordinates
        #
        r_phi_curve = N.zeros((len(curve[0,:]),2))
        for ii in N.arange(len(curve[0,:])):
            r_phi_curve[ii,0] = N.sqrt(curve[0,ii]**2 + curve[1,ii]**2 )
            r_phi_curve[ii,1] = N.arctan2(curve[0,ii],curve[1,ii])% (2*pi)

        #find index with highest r
        largest_r_idx = N.argmax(r_phi_curve[:,0])

        #check, if perhaps more values with same r - if so, take point with lowest phi
        other_idces = list(N.where(r_phi_curve[:,0]==r_phi_curve[largest_r_idx,0]))
        if len(other_idces) > 1:
            best_idx        = N.argmin(r_phi_curve[other_idces,1])
            start_idx_curve = other_idces[best_idx]
        else:
            start_idx_curve = largest_r_idx

        if not start_idx_curve == 0:
            pass
            #logger.debug( 'redefined start point to %i for curve\n'%(start_idx_curve) )

        #check orientation - want to go inwards

        start_r  = r_phi_curve[start_idx_curve,0]
        next_idx = (start_idx_curve + 1 )%len(r_phi_curve[:,0])
        prep_idx = (start_idx_curve - 1 )%len(r_phi_curve[:,0])
        next_r   = r_phi_curve[next_idx,0]

        keep_direction = True
        if next_r <= start_r:
            #check, if next R is on other side of area - look at total distance
            # if yes, reverse direction
            dist_first_next  = (curve[0,next_idx]-curve[0,start_idx_curve])**2\
                               +(curve[1,next_idx]-curve[1,start_idx_curve])**2
            dist_first_other = (curve[0,prep_idx]-curve[0,start_idx_curve])**2\
                               +(curve[1,prep_idx]-curve[1,start_idx_curve])**2

            if  dist_first_next > dist_first_other:
                keep_direction = False

        if keep_direction:
            #direction is kept

            #logger.debug( 'curve with same direction as before\n' )
            for jj in N.arange(len(curve[0,:])):
                running_idx = (start_idx_curve + jj) %len(curve[0,:])
                sorted_curve[0,jj] = curve[0,running_idx]
                sorted_curve[1,jj] = curve[1,running_idx]

        else:
            #direction  is reversed
            #logger.debug( 'curve with reverted direction\n' )
            for jj in N.arange(len(curve[0,:])):
                running_idx = (start_idx_curve - jj) %len(curve[0,:])
                sorted_curve[0,jj] = curve[0,running_idx]
                sorted_curve[1,jj] = curve[1,running_idx]


        # check if step of first to second point does not have large angle
        # step (problem caused by projection of point (pole) onto whole
        # edge - if this first angle step is larger than the one between
        # points 2 and three, correct position of first point: keep R, but
        # take angle with same difference as point 2 to point 3

        angle_point_1 = (N.arctan2(sorted_curve[0,0],sorted_curve[1,0])%(2*pi))
        angle_point_2 = (N.arctan2(sorted_curve[0,1],sorted_curve[1,1])%(2*pi))
        angle_point_3 = (N.arctan2(sorted_curve[0,2],sorted_curve[1,2])%(2*pi))

        angle_diff_23 = ( angle_point_3 - angle_point_2 )
        if angle_diff_23 > pi :
            angle_diff_23 = (-angle_diff_23)%(2*pi)

        angle_diff_12 = ( angle_point_2 - angle_point_1 )
        if angle_diff_12 > pi :
            angle_diff_12 = (-angle_diff_12)%(2*pi)

        if N.abs( angle_diff_12) > N.abs( angle_diff_23):
            r_old = N.sqrt(sorted_curve[0,0]**2 + sorted_curve[1,0]**2)
            new_angle = (angle_point_2 - angle_diff_23)%(2*pi)
            sorted_curve[0,0] = r_old * N.sin(new_angle)
            sorted_curve[1,0] = r_old * N.cos(new_angle)


        return sorted_curve


    #---------------------------------------------------------------


    def _smooth_curves(self):
        """
        Corrects curves for potential large gaps, resulting in strange
        intersection lines on nodals of round and irreagularly shaped
        areas.

        At least one coordinte point on each degree on the circle is assured.

        """

        list_of_curves_2_smooth = ['nodalline_negative', 'nodalline_positive','FP1','FP2']

        points_per_degree = self._plot_n_points/360.

        for curve2smooth in list_of_curves_2_smooth:
            obj_name = curve2smooth+'_in_order'
            obj = getattr(self,'_'+obj_name).transpose()


            smoothed_array      = N.zeros((1,2))
            smoothed_array[0,:] = obj[0]

            #now in shape (n_points,2)
            for idx,val in enumerate(obj[:-1]):
                r1   = N.sqrt(val[0]**2 + val[1]**2)
                r2   = N.sqrt(obj[idx+1][0]**2 + obj[idx+1][1]**2   )
                phi1 = N.arctan2(val[0],val[1])
                phi2 = N.arctan2(obj[idx+1][0],obj[idx+1][1])

                phi2_larger      = N.sign( phi2 - phi1 )
                angle_smaller_pi = N.sign( pi - abs(phi2 - phi1) )

                if phi2_larger * angle_smaller_pi > 0:
                    go_cw     = True
                    openangle = (phi2 - phi1)%(2*pi)
                else:
                    go_cw     = False
                    openangle = (phi1 - phi2)%(2*pi)

                openangle_deg = openangle*rad2deg
                radius_diff   = r2 -r1

                if openangle_deg > 1./points_per_degree:

                    n_fillpoints = int(openangle_deg * points_per_degree)
                    fill_array   = N.zeros((n_fillpoints,2))
                    if go_cw:
                        angles       = ((N.arange(n_fillpoints)+1)*openangle/(n_fillpoints+1) + phi1)%(2*pi)
                    else:
                        angles       = (phi1 - (N.arange(n_fillpoints)+1)*openangle/(n_fillpoints+1) )%(2*pi)

                    radii        = (N.arange(n_fillpoints)+1)*radius_diff/(n_fillpoints+1) + r1

                    fill_array[:,0] = radii * N.sin(angles)
                    fill_array[:,1] = radii * N.cos(angles)

                    smoothed_array = N.append(smoothed_array,fill_array,axis=0)

                smoothed_array = N.append(smoothed_array,[obj[idx+1]],axis=0)


            setattr(self,'_'+curve2smooth+'_final',smoothed_array.transpose())

    #---------------------------------------------------------------

    def _check_curve_in_curve(self):
        """
        Checks, if one of the two nodallines contains the other one
        completely. If so, the order of colours is re-adapted,
        assuring the correct order when doing the overlay plotting.
        """


        lo_points_in_pos_curve = list(self._nodalline_positive_final.transpose())
        lo_points_in_neg_curve = list(self._nodalline_negative_final.transpose())

        # check, if negative curve completely within positive curve
        mask_neg_in_pos = 0
        for neg_point in lo_points_in_neg_curve:
            #mask_neg_in_pos *=  self._point_inside_polygon(neg_point[0], neg_point[1],lo_points_in_pos_curve )
            mask_neg_in_pos +=  self._pnpoly(N.array(lo_points_in_pos_curve),N.array([neg_point[0], neg_point[1]]) )
        if mask_neg_in_pos > len(lo_points_in_neg_curve)-3:
            self._plot_curve_in_curve  =   1

        # check, if positive curve completely within negative curve
        mask_pos_in_neg = 0
        for pos_point in lo_points_in_pos_curve:
            mask_pos_in_neg +=  self._pnpoly(N.array(lo_points_in_neg_curve),N.array([pos_point[0], pos_point[1]]) )
        if mask_pos_in_neg > len(lo_points_in_pos_curve)-3:
            self._plot_curve_in_curve  =   -1


        #correct for ONE special case: double couple with its eigensystem = NED basis system:
        testarray = [1.,0,0,0,1,0,0,0,1]
        if N.prod(self.MT._rotation_matrix.A1 == testarray) and (self.MT._eigenvalues[1]==0 ):
            self._plot_curve_in_curve = -1
            self._plot_clr_order      =  1

    #-------------------------------------------------------------------
    #    determine if a point is inside a given polygon or not
    # Polygon is a list of (x,y) pairs.<

    def _point_inside_polygon(self,x,y,poly):

        n = len(poly)
        inside =False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside
    #---------------------------------------------------------------

    def _pnpoly(self,verts,point):
        """Check whether point is in the polygon defined by verts.

        verts - 2xN array
        point - (2,) array

        See http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
        """

        verts = verts.astype(float)
        x,y = point

        xpi = verts[:,0]
        ypi = verts[:,1]
        # shift
        xpj = xpi[N.arange(xpi.size)-1]
        ypj = ypi[N.arange(ypi.size)-1]

        possible_crossings = ((ypi <= y) & (y < ypj)) | ((ypj <= y) & (y < ypi))

        xpi = xpi[possible_crossings]
        ypi = ypi[possible_crossings]
        xpj = xpj[possible_crossings]
        ypj = ypj[possible_crossings]

        crossings = x < (xpj-xpi)*(y - ypi) / (ypj - ypi) + xpi

        return sum(crossings) % 2
    #---------------------------------------------------------------

    def _projection_2_unit_sphere(self):
        """
        Brings the complete solution (from stereographic projection)
        onto the unit sphere by just shrinking the maximum radius of
        all points to 1.


        This keeps the area definitions, so the colouring is not affected.
        """

        list_of_objects_2_project = ['nodalline_positive_final', 'nodalline_negative_final']
        lo_fps = ['FP1_final', 'FP2_final']

        for obj2proj in list_of_objects_2_project:
            obj = getattr(self,'_'+obj2proj).transpose().copy()
            for idx,val in enumerate(obj):
                old_radius = N.sqrt(val[0]**2+val[1]**2)
                if old_radius > 1:
                    obj[idx,0] = val[0]/old_radius
                    obj[idx,1] = val[1]/old_radius

            setattr(self,'_'+obj2proj+'_US',obj.transpose())

        for fp in lo_fps:
            obj = getattr(self,'_'+fp).transpose().copy()

            tmp_obj = []
            for idx,val in enumerate(obj):
                old_radius = N.sqrt(val[0]**2+val[1]**2)
                if old_radius <=  1+epsilon:
                    tmp_obj.append(val)
            tmp_obj2 = N.array(tmp_obj).transpose()
            tmp_obj3 = self._sort_curve_points(tmp_obj2)

            setattr(self,'_'+fp+'_US',tmp_obj3)

        lo_visible_EV = []

        for idx,val in enumerate(self._all_EV_2D.transpose()):
            r_ev = N.sqrt(val[0]**2 + val[1]**2)
            if r_ev <= 1:
                lo_visible_EV.append([val[0],val[1],idx])
        visible_EVs = N.array(lo_visible_EV)

        self._all_EV_2D_US = visible_EVs


        lo_visible_BV = []
        dummy_list1   = []
        direction_letters = list('NSEWDU')

        for idx,val in enumerate(self._all_BV_2D.transpose()):
            r_bv = N.sqrt(val[0]**2 + val[1]**2)
            if r_bv <= 1:
                if idx == 1 and 'N' in dummy_list1:
                    continue
                elif idx == 3 and 'E' in dummy_list1:
                    continue
                elif idx == 5 and 'D' in dummy_list1:
                    continue
                else:
                    lo_visible_BV.append([val[0],val[1],idx])
                    dummy_list1.append(direction_letters[idx])

        visible_BVs = N.array(lo_visible_BV)

        self._all_BV_2D_US = visible_BVs

   #---------------------------------------------------------------

    def _plot_US(self):
        """
        Method for generating the final plot of the beachball projection on the unit sphere.


        Additionally, the plot can be saved in a file on the fly.




        """

        import matplotlib
        import pylab as P
        from matplotlib import interactive

        plotfig = self._setup_plot_US(P)


        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile+'.'+self._plot_outfile_format, dpi=self._plot_dpi, transparent=True, format=self._plot_outfile_format)

            except:
                print 'saving of plot not possible'

        P.show()

        P.close('all')
        del P
        del matplotlib

#-------------------------------------------------------------------
    def _setup_plot_US(self,P):
        """
        Setting up the figure with the final plot of the unit sphere.

        Either called by _plot_US or by _just_save_bb
        """

        P.close(667)
        plotfig = P.figure(667,figsize=(self._plot_size,self._plot_size) )
        plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        ax = plotfig.add_subplot(111, aspect='equal')

        ax.axison = False

        neg_nodalline = self._nodalline_negative_final_US
        pos_nodalline = self._nodalline_positive_final_US
        FP1_2_plot     = self._FP1_final_US
        FP2_2_plot     = self._FP2_final_US


        US             = self._unit_sphere


        tension_colour    = self._plot_tension_colour
        pressure_colour  = self._plot_pressure_colour


        if self._plot_fill_flag:

            if self._plot_clr_order > 0 :

                ax.fill(US[0,:],US[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( neg_nodalline[0,:] ,neg_nodalline[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(US[0,:],US[1,:], fc=tension_colour , alpha= self._plot_fill_alpha*self._plot_total_alpha)

                    if self._plot_curve_in_curve < 1 :
                        ax.fill(neg_nodalline[0,:] ,neg_nodalline[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass
                    else:
                        ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:] ,fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( neg_nodalline[0,:] ,neg_nodalline[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass

                EV_sym = ['m^','b^','g^','mv','bv','gv']
                EV_labels = ['P','N','T','P','N','T']

                if self._plot_show_princ_axes:
                    for val in self._all_EV_2D_US:
                        ax.plot([val[0]],[val[1]],EV_sym[int(val[2])],ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha )

            else:
                ax.fill(US[0,:],US[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( neg_nodalline[0,:] ,neg_nodalline[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(US[0,:],US[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )

                    if self._plot_curve_in_curve < 1 :
                        ax.fill( neg_nodalline[0,:] ,neg_nodalline[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:] ,fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass
                    else:
                        ax.fill( pos_nodalline[0,:] ,pos_nodalline[1,:] ,fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( neg_nodalline[0,:] ,neg_nodalline[1,:],fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass


        EV_sym    = ['g^','b^','m^','gv','bv','mv']
        #EV_labels = ['T','N','P','T','N','P']
        if self._plot_show_princ_axes:
            for val in self._all_EV_2D_US:
                ax.plot([val[0]],[val[1]],EV_sym[int(val[2])],ms=self._plot_princ_axes_symsize,lw=self._plot_princ_axes_lw ,alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)



        #
        # set all nodallines and faultplanes for plotting:
        #

        ax.plot( neg_nodalline[0,:] ,neg_nodalline[1,:],c=self._plot_nodalline_colour,ls='-',lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha )

        ax.plot( pos_nodalline[0,:] ,pos_nodalline[1,:],c=self._plot_nodalline_colour,ls='-',lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha)



        if self._plot_show_faultplanes:

            ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha)

            ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha)

        elif self._plot_show_1faultplane:
            if not self._plot_show_FP_index in [1,2]:
                print 'no fault plane specified for being plotted... continue without faultplane'
                pass
            else:
                if self._plot_show_FP_index == 1:

                    ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha)

                else:
                    ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha)


        #if isotropic part shall be displayed, fill the circle completely with the appropriate colour
        if self._pure_isotropic:
            #f abs( N.trace( self._M )) > epsilon:
            if self._plot_clr_order < 0:
                ax.fill( US[0,:],US[1,:],fc=tension_colour, alpha= 1,zorder=100 )
            else:
                ax.fill( US[0,:],US[1,:],fc=pressure_colour, alpha= 1,zorder=100 )

        #plot outer circle line of US
        ax.plot(US[0,:],US[1,:],c=self._plot_outerline_colour,ls='-',lw=self._plot_outerline_width,alpha= self._plot_outerline_alpha*self._plot_total_alpha )


        #plot NED basis vectors
        if self._plot_show_basis_axes:

            plot_size_in_points = self._plot_size * 2.54 * 72
            points_per_unit = plot_size_in_points/2.

            fontsize  = plot_size_in_points / 40.
            symsize   = plot_size_in_points / 61.

            direction_letters = list('NSEWDU')
            #print direction_letters
            for val in self._all_BV_2D_US:
                #print val
                x_coord = val[0]
                y_coord = val[1]
                np_letter = direction_letters[int(val[2])]

                rot_angle    = - N.arctan2(y_coord,x_coord) + pi/2.
                original_rho = N.sqrt(x_coord**2 + y_coord**2)

                marker_x  = ( original_rho - ( 1.5* symsize  / points_per_unit ) ) * N.sin( rot_angle )
                marker_y  = ( original_rho - ( 1.5* symsize  / points_per_unit ) ) * N.cos( rot_angle )
                annot_x   = ( original_rho - ( 4.5* fontsize / points_per_unit ) ) * N.sin( rot_angle )
                annot_y   = ( original_rho - ( 4.5* fontsize / points_per_unit ) ) * N.cos( rot_angle )

                ax.text(annot_x,annot_y,np_letter,horizontalalignment='center', size=fontsize,weight='bold', verticalalignment='center',\
                        bbox=dict(edgecolor='white',facecolor='white', alpha=1))

                if original_rho > epsilon:
                    ax.scatter([marker_x],[marker_y],marker=(3,0,rot_angle) ,s=symsize**2,c='k',facecolor='k',zorder=300)
                else:
                    ax.scatter([x_coord],[y_coord],marker=(4,1,rot_angle) ,s=symsize**2,c='k',facecolor='k',zorder=300)



        #plot 4 transparent fake points, guaranteeing full visibilty of the sphere in the plot
        ax.plot([0,1.05,0,-1.05],[1.05,0,-1.05,0],',',alpha=0.)

        #scaling behaviour
        ax.autoscale_view(tight=True, scalex=True, scaley=True)


        return plotfig





#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#
#
#  input and call management, if run from the shell !!
#
#
#-------------------------------------------------------------------
#-------------------------------------------------------------------

if __name__ == "__main__":

    import os
    import os.path as op
    from optparse import OptionParser,OptionGroup

    decomp_attrib_map_keys = ('in','out','type',
                                'full',
                                'iso','iso_perc',
                                'dev','devi','devi_perc',
                                'dc','dc_perc',
                                'dc2','dc2_perc',
                                'dc3','dc3_perc',
                                'clvd','clvd_perc',
                                'mom','mag',
                                'eigvals','eigvecs',
                                't','n','p')

    decomp_attrib_map = dict(zip( decomp_attrib_map_keys,
                                ('input_system','output_system','decomp_type',
                                'M',
                                'iso','iso_percentage',
                                'devi','devi','devi_percentage',
                                'DC','DC_percentage',
                                'DC2','DC2_percentage',
                                'DC3','DC3_percentage',
                                'CLVD','CLVD_percentage',
                                'moment','mag',
                                'eigvals','eigvecs',
                                't_axis','null_axis','p_axis')
                            ))

    lo_allowed_systems     = ['NED','USE','XYZ','NWU']


    #------------------------------------------------------------------

    def _handle_input(call, M_in, call_args,optparser):
        """
        take the original method and its arguments, the source mechanism,
        and the dictionary with proper parsers for each call,
        """

        #construct a dict with  consistent keywordargs suited for the current call
        kwargs = _parse_arguments(call, call_args, optparser)

        #set the fitting input basis system
        in_system = kwargs.get('in_system','NED')
        out_system = kwargs.get('out_system', 'NED')

        #build the moment tensor object
        mt     = MomentTensor(M=M_in, in_system=in_system, out_system=out_system)

        # if only parts of M are to be plotted, M must be reduced to this part already here!
        if  call == 'plot' and  kwargs['plot_part_of_m']:
            if kwargs['plot_part_of_m'] == 'iso':
                mt     = MomentTensor(M=mt.get_iso(), in_system=in_system, out_system=out_system)

            if kwargs['plot_part_of_m'] == 'devi':
                mt     = MomentTensor(M=mt.get_devi(), in_system=in_system, out_system=out_system)

            if kwargs['plot_part_of_m'] == 'dc':
                mt     = MomentTensor(M=mt.get_DC(), in_system=in_system, out_system=out_system)

            if kwargs['plot_part_of_m'] == 'clvd':
                mt     = MomentTensor(M=mt.get_CLVD(), in_system=in_system, out_system=out_system)



        #call the main routine to handle the moment tensor
        return  _call_main(mt,call,kwargs)

    #------------------------------------------------------------------
    def _call_main(MT,main_call,kwargs_dict ):

        if main_call == 'plot':
           return  _call_plot(MT,kwargs_dict)

        elif main_call == 'gmt':
           return  _call_gmt(MT,kwargs_dict)

        elif main_call == 'decompose':
           return  _call_decompose(MT,kwargs_dict)

        elif main_call == 'describe':
           return  _call_describe(MT,kwargs_dict)


    #------------------------------------------------------------------
    def _call_plot(MT,kwargs_dict):


        bb2plot = BeachBall(MT,kwargs_dict)

        if kwargs_dict['plot_save_plot']:
            bb2plot.save_BB(kwargs_dict)
            return

        #import pylab as P

        if kwargs_dict['plot_pa_plot']:
            bb2plot.pa_plot(kwargs_dict)
            return

        if kwargs_dict['plot_full_sphere']:
            bb2plot.full_sphere_plot(kwargs_dict)
            return

        bb2plot.ploBB(kwargs_dict)

        return

    def _call_gmt(MT,kwargs_dict):
        bb = BeachBall(MT,kwargs_dict)
        return bb.get_psxy(kwargs_dict)

    #------------------------------------------------------------------

    def _call_decompose(MT,kwargs_dict):

        MT._isotropic         = None
        MT._deviatoric        = None
        MT._DC                = None
        MT._iso_percentage    = None
        MT._DC_percentage     = None
        MT._DC2               = None
        MT._DC3               = None
        MT._DC2_percentage    = None
        MT._CLVD              = None
        MT._seismic_moment    = None
        MT._moment_magnitude  = None



        out_system = kwargs_dict['out_system']
        MT._output_basis  = out_system
        MT._decomposition_key = kwargs_dict['decomposition_key']

        MT._decompose_M()

        print

        #build argument for local call within MT object:
        lo_args = kwargs_dict['decomp_out_part']

        if not lo_args:
            lo_args = decomp_attrib_map_keys

        #for list of elements:
        for arg in lo_args:
            print getattr(MT,'get_'+decomp_attrib_map[arg])(style='y',system=out_system )


    #------------------------------------------------------------------

    def _call_describe(MT, kwargs_dict):
        print MT


    def _build_gmt_dict(options,optparser):

        consistent_kwargs_dict = {}
        temp_dict              = {}
        lo_allowed_options     = ['GMT_string_type','GMT_scaling','GMT_tension_colour','GMT_pressure_colour','GMT_show_2FP2','GMT_show_1FP','plot_viewpoint','GMT_plot_isotropic_part','GMT_projection']

        #check for allowed options:
        for ao in lo_allowed_options:
            if hasattr(options,ao):
                temp_dict[ao] = getattr(options,ao)

        if temp_dict['GMT_show_1FP']:
            try:
                if int(float(temp_dict['GMT_show_1FP'])) in  [1,2]:

                    consistent_kwargs_dict['_GMT_1fp'] = int(float(temp_dict['GMT_show_1FP']))

            except:
                pass

        if temp_dict['GMT_show_2FP2']:
            temp_dict['GMT_show_1FP'] = 0

            consistent_kwargs_dict['_GMT_2fps']   = True
            consistent_kwargs_dict['_GMT_1fp']    = 0

        if temp_dict['GMT_string_type'][0].lower() not in ['f','l','e']:
            print 'type of requested string not known - taking "fill" instead'
            consistent_kwargs_dict['_GMT_type']          = 'fill'

        else:
            if temp_dict['GMT_string_type'][0] == 'f':
                consistent_kwargs_dict['_GMT_type']          = 'fill'
            elif temp_dict['GMT_string_type'][0] == 'l':
                consistent_kwargs_dict['_GMT_type']          = 'lines'
            else:
                consistent_kwargs_dict['_GMT_type']          = 'EVs'

        if float(temp_dict['GMT_scaling']) < epsilon:
            print 'GMT scaling factor must be a factor larger than %f - set to 1, due to obviously stupid input value'%epsilon
            temp_dict['GMT_scaling'] = 1


        if temp_dict['plot_viewpoint']:
            try:
                vp = temp_dict['plot_viewpoint'].split(',')
                if not len(vp) == 3:
                    raise
                if not -90<=float(vp[0])<=90:
                    raise
                if not -180<=float(vp[1])<=180:
                    raise
                if not 0<= float(vp[2])%360<=360:
                    raise
                consistent_kwargs_dict['plot_viewpoint'] = [float(vp[0]),float(vp[1]),float(vp[2])]
            except:
                #print 'argument of "-V" must be of form "lat,lon,azi" with lat=[-90,90], lon=[-180,180], azi=[0,360]'
                pass

        if temp_dict['GMT_projection']:
            lo_allowed_projections = ['stereo','ortho','lambert']#,'gnom']
            do_allowed_projections = dict(zip(('s','o','l','g'), ('stereo','ortho','lambert','gnom')))
            try:
                if temp_dict['GMT_projection'].lower() in lo_allowed_projections:
                    consistent_kwargs_dict['plot_projection'] = temp_dict['GMT_projection'].lower()
                elif temp_dict['GMT_projection'].lower() in  do_allowed_projections.keys():
                    consistent_kwargs_dict['plot_projection'] = do_allowed_projections[ temp_dict['GMT_projection'].lower()]
                else:
                    consistent_kwargs_dict['plot_projection'] = 'stereo'
            except:
                pass




        consistent_kwargs_dict['_GMT_scaling']             = temp_dict['GMT_scaling']
        consistent_kwargs_dict['_GMT_tension_colour']         = temp_dict['GMT_tension_colour']
        consistent_kwargs_dict['_GMT_pressure_colour']       = temp_dict['GMT_pressure_colour']
        consistent_kwargs_dict['_plot_isotropic_part'] = temp_dict['GMT_plot_isotropic_part']

        return consistent_kwargs_dict

    #------------------------------------------------------------------

    def _build_decompose_dict(options,optparser):

        consistent_kwargs_dict = {}
        temp_dict              = {}
        lo_allowed_options     = ['decomp_out_complete','decomp_out_fancy','decomp_out_part','in_system','out_system','decomp_key']

        #check for allowed options:
        for ao in lo_allowed_options:
            if hasattr(options,ao):
                temp_dict[ao] = getattr(options,ao)

        for k in 'in_system', 'out_system':
            s = getattr(options,k).upper()
            if s not in lo_allowed_systems:
                sys.exit('Unavailable coordinate system: %s' % s)

            consistent_kwargs_dict[k] = s

        consistent_kwargs_dict['decomposition_key'] = int(temp_dict['decomp_key'])

        if temp_dict['decomp_out_part'] is None:
            consistent_kwargs_dict['decomp_out_part'] = None
        else:
            parts = [ x.strip().lower() for x in temp_dict['decomp_out_part'].split(',') ]
            for part in parts:
                if part not in decomp_attrib_map_keys:
                    sys.exit('Unavailable decomposition part: %s' % part)

            consistent_kwargs_dict['decomp_out_part'] = parts

        consistent_kwargs_dict['style'] = 'y'

        return consistent_kwargs_dict


    def _build_plot_dict(options,optparser):
        consistent_kwargs_dict = {}
        temp_dict              = {}

        lo_allowed_options     = ['plot_outfile','plot_pa_plot','plot_full_sphere','plot_part_of_m',\
                                  'plot_viewpoint','plot_projection','plot_show_upper_hemis','plot_n_points','plot_size',\
                                  'plot_tension_colour','plot_pressure_colour','plot_total_alpha','plot_show_faultplanes',\
                                  'plot_show_1faultplane','plot_show_princ_axes','plot_show_basis_axes','plot_outerline',\
                                  'plot_nodalline','plot_dpi','plot_only_lines','plot_input_system','plot_isotropic_part']
        #check for allowed options:
        for ao in lo_allowed_options:
            if hasattr(options,ao):
                temp_dict[ao] = getattr(options,ao)


        consistent_kwargs_dict['plot_save_plot'] = False
        if temp_dict['plot_outfile']:
            consistent_kwargs_dict['plot_save_plot'] = True
            lo_possible_formats = ['svg','png','eps','pdf','ps']

            try:
                (filepath, filename)   = op.split( temp_dict['plot_outfile'])
                if not filename:
                    filename = 'dummy_filename.svg'
                (shortname, extension) = op.splitext(filename)
                if not shortname:
                    shortname = 'dummy_shortname'


                if extension[1:].lower() in lo_possible_formats:
                    consistent_kwargs_dict['plot_outfile_format'] = extension[1:].lower()

                    if shortname.endswith('.'):
                        consistent_kwargs_dict['plot_outfile']        = op.realpath(op.abspath(op.join(os.curdir,filepath,shortname+extension[1:].lower() )))

                    else:
                        consistent_kwargs_dict['plot_outfile']        = op.realpath(op.abspath(op.join(os.curdir,filepath,shortname+'.'+extension[1:].lower() )))
                else:
                    if filename.endswith('.'):
                        consistent_kwargs_dict['plot_outfile']        = op.realpath(op.abspath(op.join(os.curdir,filepath,filename+lo_possible_formats[0])))
                    else:
                        consistent_kwargs_dict['plot_outfile']        = op.realpath(op.abspath(op.join(os.curdir,filepath,filename+'.'+lo_possible_formats[0])))
                    consistent_kwargs_dict['plot_outfile_format'] = lo_possible_formats[0]

            except:
                exit('please provide valid filename: <name>.<format>  !!\n  <format> must be svg, png, eps, pdf, or ps ')


        if temp_dict['plot_pa_plot']:
            consistent_kwargs_dict['plot_pa_plot']     = True
        else:
            consistent_kwargs_dict['plot_pa_plot']     = False

        #
        #
        if temp_dict['plot_full_sphere']:
            consistent_kwargs_dict['plot_full_sphere'] = True
            consistent_kwargs_dict['plot_pa_plot']     = False
        else:
            consistent_kwargs_dict['plot_full_sphere'] = False

        #
        #
        if temp_dict['plot_part_of_m']:
            try:
                plottable_part_raw = temp_dict['plot_part_of_m'].lower()[:2]
                if plottable_part_raw == 'is':
                    plottable_part = 'iso'
                elif plottable_part_raw == 'de':
                    plottable_part = 'devi'
                elif plottable_part_raw == 'dc':
                    plottable_part = 'dc'
                elif plottable_part_raw == 'cl':
                    plottable_part = 'clvd'
                else:
                   plottable_part = False

                consistent_kwargs_dict['plot_part_of_m'] = plottable_part

            except:
                consistent_kwargs_dict['plot_part_of_m'] = False

        else:
            consistent_kwargs_dict['plot_part_of_m'] = False


        #
        #
        if temp_dict['plot_viewpoint']:
            try:
                vp = temp_dict['plot_viewpoint'].split(',')
                if not len(vp) == 3:
                    raise
                if not -90<=float(vp[0])<=90:
                    raise
                if not -180<=float(vp[1])<=180:
                    raise
                if not 0<= float(vp[2])%360<=360:
                    raise
                consistent_kwargs_dict['plot_viewpoint'] = [float(vp[0]),float(vp[1]),float(vp[2])]
            except:
                pass


        #
        #
        if temp_dict['plot_projection']:
            lo_allowed_projections = ['stereo','ortho','lambert']#,'gnom']
            do_allowed_projections = dict(zip(('s','o','l','g'), ('stereo','ortho','lambert','gnom')))
            try:
                if temp_dict['plot_projection'].lower() in lo_allowed_projections:
                    consistent_kwargs_dict['plot_projection'] = temp_dict['plot_projection'].lower()
                elif temp_dict['plot_projection'].lower() in  do_allowed_projections.keys():
                    consistent_kwargs_dict['plot_projection'] = do_allowed_projections[ temp_dict['plot_projection'].lower()]
                else:
                    consistent_kwargs_dict['plot_projection'] = 'stereo'
            except:
                pass


        #
        #
        if temp_dict['plot_show_upper_hemis']:
            consistent_kwargs_dict['plot_show_upper_hemis'] = True

        #
        #
        if temp_dict['plot_n_points']:
            try:
                if temp_dict['plot_n_points']>360:
                    consistent_kwargs_dict['plot_n_points'] = int(temp_dict['plot_n_points'])
            except:
                pass

        #
        #
        if temp_dict['plot_size']:
            try:
                if  0.01 < temp_dict['plot_size'] <= 1:
                    consistent_kwargs_dict['plot_size'] = temp_dict['plot_size']*10/2.54
                elif 1 < temp_dict['plot_size'] < 45:
                    consistent_kwargs_dict['plot_size'] = temp_dict['plot_size']/2.54
                else:
                    consistent_kwargs_dict['plot_size'] = 5

                #
                #
                consistent_kwargs_dict['plot_aux_plot_size'] = consistent_kwargs_dict['plot_size']

            except:
                pass

        #
        #
        if temp_dict['plot_pressure_colour']:
            try:
                sec_colour_raw = temp_dict['plot_pressure_colour'].split(',')
                if len(sec_colour_raw) == 1 :
                    if sec_colour_raw[0].lower()[0] in list('bgrcmykw'):
                       consistent_kwargs_dict['plot_pressure_colour'] = sec_colour_raw[0].lower()[0]
                    else:
                        raise
                elif len(sec_colour_raw)==3:
                    for sc in sec_colour_raw:
                        if not 0<= (int(sc))<=255:
                            raise
                    consistent_kwargs_dict['plot_pressure_colour'] = (float(sec_colour_raw[0])/255.,float(sec_colour_raw[1])/255., float(sec_colour_raw[2])/255.)

                else:
                    raise

            except:
                pass

        #
        ##
        if temp_dict['plot_tension_colour']:
            try:
                sec_colour_raw = temp_dict['plot_tension_colour'].split(',')
                if len(sec_colour_raw)==1:
                    if sec_colour_raw[0].lower()[0] in list('bgrcmykw'):
                       consistent_kwargs_dict['plot_tension_colour'] = sec_colour_raw[0].lower()[0]
                    else:
                        raise
                elif len(sec_colour_raw)==3:
                    for sc in sec_colour_raw:
                        if not 0<= (int(float(sc)))<=255:
                            raise

                    consistent_kwargs_dict['plot_tension_colour'] = (float(sec_colour_raw[0])/255.,float(sec_colour_raw[1])/255., float(sec_colour_raw[2])/255.)

                else:
                    raise

            except:
                pass
        #
        #
        if temp_dict['plot_total_alpha']:
            try:
                if not 0<=float(temp_dict['plot_total_alpha']) <= 1:
                    consistent_kwargs_dict['plot_total_alpha'] = 1
                else:
                    consistent_kwargs_dict['plot_total_alpha'] = float(temp_dict['plot_total_alpha'])

            except:
                pass

        #
        #
        if temp_dict['plot_show_1faultplane']:
            consistent_kwargs_dict['plot_show_1faultplane'] = True
            try:
                fp_args = temp_dict['plot_show_1faultplane']

                if not int(fp_args[0]) in [1,2]:
                    consistent_kwargs_dict['plot_show_FP_index'] = 1
                else:
                    consistent_kwargs_dict['plot_show_FP_index'] = int(fp_args[0])

                if not 0 < float(fp_args[1]) <= 20:
                    consistent_kwargs_dict['plot_faultplane_width'] = 2
                else:
                    consistent_kwargs_dict['plot_faultplane_width'] = float(fp_args[1])


                try:
                    sec_colour_raw = fp_args[2].split(',')
                    if len(sec_colour_raw)==1:
                        if sec_colour_raw[0].lower()[0] in list('bgrcmykw'):
                            consistent_kwargs_dict['plot_faultplane_colour'] = sec_colour_raw[0].lower()[0]
                        else:
                            raise
                    elif len(sec_colour_raw)==3:
                        for sc in sec_colour_raw:
                            if not 0<= (int(sc))<=255:
                                raise
                        consistent_kwargs_dict['plot_faultplane_colour'] = (float(sec_colour_raw[0])/255.,float(sec_colour_raw[1])/255., float(sec_colour_raw[2])/255.)

                    else:
                        raise

                except:
                    consistent_kwargs_dict['plot_faultplane_colour'] = 'k'

                try:
                    if  0<= float(fp_args[3]) <= 1:
                        consistent_kwargs_dict['plot_faultplane_alpha'] = float(fp_args[3])
                except:
                    consistent_kwargs_dict['plot_faultplane_alpha'] = 1


            except:
                pass

        #
        #
        if temp_dict['plot_show_faultplanes']:
            consistent_kwargs_dict['plot_show_faultplanes'] = True
            consistent_kwargs_dict['plot_show_1faultplane'] = False

        #
        #
        if temp_dict['plot_dpi']:
            try:
                if 200 <= int(temp_dict['plot_dpi']) <= 2000:
                    consistent_kwargs_dict['plot_dpi'] = int(temp_dict['plot_dpi'])
                else:
                    raise
            except:
                pass


        #
        #
        if temp_dict['plot_only_lines']:
            consistent_kwargs_dict['plot_fill_flag'] = False

        #
        #
        if temp_dict['plot_outerline']:
            consistent_kwargs_dict['plot_outerline'] = True
            try:
                fp_args = temp_dict['plot_outerline']

                if not 0 < float(fp_args[0]) <= 20:
                    consistent_kwargs_dict['plot_outerline_width'] = 2
                else:
                    consistent_kwargs_dict['plot_outerline_width'] = float(fp_args[0])


                try:
                    sec_colour_raw = fp_args[1].split(',')
                    if len(sec_colour_raw)==1:
                        if sec_colour_raw[0].lower()[0] in list('bgrcmykw'):
                            consistent_kwargs_dict['plot_outerline_colour'] = sec_colour_raw[0].lower()[0]
                        else:
                            raise
                    elif len(sec_colour_raw)==3:
                        for sc in sec_colour_raw:
                            if not 0<= (int(sc))<=255:
                                raise
                        consistent_kwargs_dict['plot_outerline_colour'] = (float(sec_colour_raw[0])/255.,float(sec_colour_raw[1])/255., float(sec_colour_raw[2])/255.)

                    else:
                        raise

                except:
                    consistent_kwargs_dict['plot_outerline_colour'] = 'k'

                try:
                    if  0<= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_outerline_alpha'] = float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_outerline_alpha'] = 1


            except:
                pass
        #
        #
        if temp_dict['plot_nodalline']:
            consistent_kwargs_dict['plot_nodalline'] = True
            try:
                fp_args = temp_dict['plot_nodalline']

                if not 0 < float(fp_args[0]) <= 20:
                    consistent_kwargs_dict['plot_nodalline_width'] = 2
                else:
                    consistent_kwargs_dict['plot_nodalline_width'] = float(fp_args[0])


                try:
                    sec_colour_raw = fp_args[1].split(',')
                    if len(sec_colour_raw)==1:
                        if sec_colour_raw[0].lower()[0] in list('bgrcmykw'):
                            consistent_kwargs_dict['plot_nodalline_colour'] = sec_colour_raw[0].lower()[0]
                        else:
                            raise
                    elif len(sec_colour_raw)==3:
                        for sc in sec_colour_raw:
                            if not 0<= (int(sc))<=255:
                                raise
                        consistent_kwargs_dict['plot_nodalline_colour'] = (float(sec_colour_raw[0])/255.,float(sec_colour_raw[1])/255., float(sec_colour_raw[2])/255.)

                    else:
                        raise

                except:
                    consistent_kwargs_dict['plot_nodalline_colour'] = 'k'

                try:
                    if  0<= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_nodalline_alpha'] = float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_nodalline_alpha'] = 1


            except:
                pass
        #
        #
        if temp_dict['plot_show_princ_axes']:
            consistent_kwargs_dict['plot_show_princ_axes'] = True
            try:
                fp_args = temp_dict['plot_show_princ_axes']

                if not 0 < float(fp_args[0]) <= 40:
                    consistent_kwargs_dict['plot_princ_axes_symsize'] = 10
                else:
                    consistent_kwargs_dict['plot_princ_axes_symsize'] = float(fp_args[0])

                if not 0 < float(fp_args[1]) <= 20:
                    consistent_kwargs_dict['plot_princ_axes_lw '] = 3
                else:
                    consistent_kwargs_dict['plot_princ_axes_lw '] = float(fp_args[1])


                try:
                    if  0<= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_princ_axes_alpha'] = float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_princ_axes_alpha'] = 1

            except:
                pass

        if temp_dict['plot_show_basis_axes']:
            consistent_kwargs_dict['plot_show_basis_axes'] = True

        if temp_dict['plot_input_system']:
            try:
                if temp_dict['plot_input_system'][:3].upper() in lo_allowed_systems:
                    consistent_kwargs_dict['in_system'] =  temp_dict['plot_input_system'][:3].upper()
                else:
                    raise
            except:
                pass

        if temp_dict['plot_isotropic_part']:
            consistent_kwargs_dict['plot_isotropic_part'] =  temp_dict['plot_isotropic_part']


        return consistent_kwargs_dict

    def _build_describe_dict(options,optparser):
        consistent_kwargs_dict = {}

        for k in 'in_system', 'out_system':
            s = getattr(options,k).upper()
            if s not in lo_allowed_systems:
                sys.exit('Unavailable coordinate system: %s' % s)

            consistent_kwargs_dict[k] = s

        return consistent_kwargs_dict


    #------------------------------------------------------------------

    def _parse_arguments(main_call, its_arguments, optparser):


        (options, args) = optparser.parse_args(its_arguments)

        #TODO check: if arguments do not start with "-" - if so, there is a lack of arguments for the previous option
        for  val2check in options.__dict__.values():
            if str(val2check).startswith('-'):
                try:
                    val2check_split = val2check.split(',')
                    for ii in val2check_split:
                        float(ii)
                except:
                    sys.exit('\n   ERROR - check carefully number of arguments for all options\n')

        if main_call =='plot':
            consistent_kwargs_dict = _build_plot_dict(options,optparser)

        elif main_call =='gmt':
            consistent_kwargs_dict = _build_gmt_dict(options,optparser)

        elif main_call =='decompose':
            consistent_kwargs_dict = _build_decompose_dict(options,optparser)

        elif main_call =='describe':
            consistent_kwargs_dict = _build_describe_dict(options,optparser)

        return consistent_kwargs_dict


    def _add_group_system(parent):
        group_system  =  OptionGroup(parent,'Basis systems')
        group_system.add_option('-i', '--input-system',
                                action="store",
                                dest='in_system',
                                metavar='<basis>',
                                default='NED',
                                help='''Define the coordinate system of the source mechanism [Default: NED].

Available coordinate systems:

    * NED: North, East, Down
    * USE: Up, South, East    (Global CMT)
    * XYZ: East, North, Up    (Jost and Herrmann)
    * NWU: North, West, Up    (Stein and Wysession)
''')

        group_system.add_option('-o', '--output-system',
                                action="store",
                                dest='out_system',
                                metavar='<basis>',
                                default='NED',
                                help="Define the coordinate system of the output. See '--input-system' for a list of available coordinate systems [Default: NED].")

        parent.add_option_group(group_system)


    #------------------------------------------------------------------
    # build dictionary with 4 (5 incl. 'save') sets of options, belonging to the 4 (5) possible calls
    #
    #

    def _build_optparsers():


        _do_parsers = {}

        desc="""
Generate a beachball representation which can be plotted with GMT.

This tool produces output which can be fed into the GMT command `psxy`. The
output consists of coordinates which describe the lines of the beachball in
standard cartesian coordinates, centered at zero.

In order to generate a beachball diagram, this tool has to be called twice
with different arguments of the --type option. First to define the colored areas
(--type=fill) and second for the nodal and border lines (--type=lines).

Example:

    mopad gmt 30,60,90 --type=fill | psxy -Jx4/4 -R-2/2/-2/2 -P -Cpsxy_fill.cpt -M -L -K > out.ps

    mopad gmt 30,60,90 --type=lines | psxy -Jx4/4 -R-2/2/-2/2 -P -Cpsxy_lines.cpt -W2p -P -M -O >> out.ps

"""

        parser_gmt            = OptionParser(usage="mopad.py gmt <source-mechanism> [options]",description=desc, formatter=MopadHelpFormatter())

        group_type            =  OptionGroup(parser_gmt,'Output')
        group_show            =  OptionGroup(parser_gmt,'Appearance')
        group_geo             =  OptionGroup(parser_gmt,'Geometry')

        group_type.add_option('-t', '--type',\
                              type='string',\
                              dest='GMT_string_type',\
                              action='store',\
                              default='fill',\
                              help='Chosing the respective psxy data set: area to fill (fill), nodal lines (lines), or eigenvector positions (ev) [Default: fill]',\
                              metavar='<type>')

        group_show.add_option('-s', '--scaling',\
                              dest='GMT_scaling',\
                              action='store',\
                              default='1',\
                              type='float',\
                              metavar='<scaling factor>',\
                              help='Spatial scaling factor of the beachball [Default: 1]')
        group_show.add_option('-r', '--colour1',\
                              dest='GMT_tension_colour',\
                              type='int',\
                              action='store',\
                              metavar='<tension colour>',\
                              default='1',\
                              help="-Z option's key (see help for 'psxy') for the tension colour of the beachball - type: integer [Default: 1]")
        group_show.add_option('-w', '--colour2',\
                              dest='GMT_pressure_colour',\
                              type='int',\
                              action='store',\
                              metavar='<pressure colour>',\
                              default='0',\
                              help="-Z option's key (see help for 'psxy') for the pressure colour of the beachball - type: integer [Default: 0]")
        group_show.add_option('-D', '--faultplanes',\
                              dest='GMT_show_2FP2',\
                              action='store_true',\
                              default=False,\
                              help='Key, if 2 faultplanes shall be shown [Default: deactivated]')
        group_show.add_option('-d', '--show_1fp',\
                              type='choice',\
                              dest='GMT_show_1FP',\
                              choices=['1', '2'],\
                              metavar='<FP index>',\
                              action='store',\
                              default=False,\
                              help='Key for plotting 1 specific faultplane - value: 1,2 [Default: None]')
        group_geo.add_option('-v', '--viewpoint',\
                              action="store",\
                              dest='plot_viewpoint',\
                              metavar='<lat,lon,azi>',\
                              default=None,\
                              help='Coordinates (in degrees) of the viewpoint onto the projection - type: comma separated 3-tuple [Default: None]')
        group_geo.add_option('-p', '--projection',\
                              action="store",\
                              dest='GMT_projection',\
                              metavar='<projection>',\
                              default=None,\
                              help='Two-dimensional projection of the sphere - value: (s)tereographic, (l)ambert, (o)rthographic [Default: (s)tereographic]')
        group_show.add_option('-I', '--show_isotropic_part',\
                              dest='GMT_plot_isotropic_part',\
                              action='store_true',\
                              default=False,\
                              help='Key for considering the isotropic part for plotting [Default: deactivated]')

        parser_gmt.add_option_group(group_type)
        parser_gmt.add_option_group(group_show)
        parser_gmt.add_option_group(group_geo)

        _do_parsers['gmt']       = parser_gmt


        ## plot
        desc_plot="""
        Plot a beachball diagram of the provided mechanism.

        Several styles and configurations are available. Also saving
        on the fly can be enabled.
        ONLY THE DEVIATORIC COMPONENT WILL BE PLOTTED by default;
        for including the isotropic part, use the '--show_isotropic_part' option!
        """
        parser_plot              = OptionParser(usage="mopad.py plot <source-mechanism> [options]",description=desc_plot,formatter=MopadHelpFormatter())

        group_save               =  OptionGroup(parser_plot,'Saving')
        group_type               =  OptionGroup(parser_plot,'Type of plot')
        group_quality            =  OptionGroup(parser_plot,'Quality')
        group_colours            =  OptionGroup(parser_plot,'Colours')
        group_misc               =  OptionGroup(parser_plot,'Miscellaneous')
        group_dc                 =  OptionGroup(parser_plot,'Fault planes')
        group_geo                =  OptionGroup(parser_plot,'Geometry')
        group_app                =  OptionGroup(parser_plot,'Appearance')



        group_save.add_option('-f', '--output_file',\
                              action="store",\
                              dest='plot_outfile',\
                              metavar='<filename>',\
                              default=None,\
                              nargs = 1,\
                              help='(Absolute) filename for saving [Default: None]')


        group_type.add_option('-E', '--eigen_system',\
                              action="store_true",\
                              dest='plot_pa_plot',\
                              default=False,\
                              help='Key for plotting principal axis system/eigensystem [Default: deactivated]')

        group_type.add_option('-O', '--full_sphere',\
                              action="store_true",\
                              dest='plot_full_sphere',\
                              default=False,\
                              help='Key for plotting the full sphere [Default: deactivated]')

        group_type.add_option('-P', '--partial',\
                              action="store",\
                              dest='plot_part_of_m',\
                              metavar='<part of M>',\
                              default=None,\
                              help='Key for plotting only a specific part of M - values: iso,devi,dc,clvd [Default: None] ')

        group_geo.add_option('-v', '--viewpoint',\
                             action="store",\
                             dest='plot_viewpoint',\
                             metavar='<lat,lon,azi>',\
                             default=None,\
                             help='Coordinates (in degrees) of the viewpoint onto the projection - type: comma separated 3-tuple [Default: None]')

        group_geo.add_option('-p', '--projection',\
                             action="store",\
                             dest='plot_projection',\
                             metavar='<projection>',\
                             default=None,\
                             help='Two-dimensional projection of the sphere - value: (s)tereographic, (l)ambert, (o)rthographic [Default: (s)tereographic]')

        group_type.add_option('-U', '--upper',\
                              action="store_true",\
                              dest='plot_show_upper_hemis',\
                              default=False,\
                              help='Key for plotting the upper hemisphere [Default: deactivated]')

        group_quality.add_option('-N', '--points',\
                                 action="store",\
                                 metavar='<no. of points>',\
                                 dest='plot_n_points',\
                                 type = "int",\
                                 default=None,\
                                 help='Minimum number of points, used for nodal lines [Default: None]')

        group_app.add_option('-s', '--size',\
                             action="store",\
                             dest='plot_size',\
                             metavar='<size in cm>',\
                             type="float",\
                             default=None,\
                             help='Size of plot (diameter) in cm [Default: None]')

        group_colours.add_option('-w', '--pressure_colour',\
                                 action="store",\
                                 dest='plot_pressure_colour',\
                                 metavar='<colour>',\
                                 default=None,\
                                 help='Colour of the tension area - values: comma separated RGB 3-tuples OR MATLAB conform colour names [Default: None]')

        group_colours.add_option('-r', '--tension_colour',\
                                 action="store",\
                                 dest='plot_tension_colour',\
                                 metavar='<colour>',\
                                 default=None,\
                                 help='Colour of the pressure area values: comma separated RGB 3-tuples OR MATLAB conform colour names [Default: None]')

        group_app.add_option('-a', '--alpha',\
                             action="store",\
                             dest='plot_total_alpha',\
                             metavar='<alpha>',\
                             type='float',\
                             default=None,\
                             help='Alpha value for the total plot - value: float between 1=opaque to 0=transparent [Default: None]')

        group_dc.add_option('-D', '--dc',\
                            action="store_true",\
                            dest='plot_show_faultplanes',\
                            default= False,\
                            help='Key for plotting both double couple faultplanes (blue) [Default: deactivated]')

        group_dc.add_option('-d', '--show1fp',\
                            action="store",\
                            metavar='<index> <linewidth> <colour> <alpha>',\
                            dest='plot_show_1faultplane',\
                            default=None,\
                            nargs = 4,\
                            help= 'Key for plotting 1 specific faultplane - 4 arguments as space separated list - index values: 1,2, linewidth value: float, line colour value: string or RGB-3-tuple, alpha value: float between 0 and 1 [Default: None] ')

        group_misc.add_option('-e', '--eigenvectors',\
                              action="store",\
                              dest='plot_show_princ_axes',\
                              metavar='<size> <linewidth> <alpha>',\
                              default=None,\
                              nargs = 3,\
                              help='Key for showing eigenvectors - 3 arguments as space separated list - symbol size value: float, symbol linewidth value: float, symbol alpha value: float between 0 and 1 [Default: None]')

        group_misc.add_option('-b', '--basis_vectors',\
                              action="store_true",\
                              dest='plot_show_basis_axes',\
                              default=False,\
                              help='Key for showing NED basis axes in plot [Default: deactivated]')

        group_app.add_option('-l', '--lines',\
                             action="store",\
                             dest='plot_outerline',\
                             metavar='<linewidth> <colour> <alpha>',\
                             nargs = 3,\
                             default=None,\
                             help='Define the style of the outer line - 3 arguments as space separated list - linewidth value: float, line colour value: string or RGB-3-tuple), alpha value: float between 0 and 1 [Default: None]')

        group_app.add_option('-n', '--nodals',\
                             action="store",\
                             dest='plot_nodalline',\
                             metavar='<linewidth> <colour> <alpha>',\
                             default=None,\
                             nargs = 3,\
                             help='Define the style of the nodal lines - 3 arguments as space separated list - linewidth value: float, line colour value: string or RGB-3-tuple), alpha value: float between 0 and 1 [Default: None]')

        group_quality.add_option('-Q', '--quality',\
                                 action="store",\
                                 dest='plot_dpi',\
                                 metavar='<dpi>',\
                                 type="int",\
                                 default=None,\
                                 help='Set the quality for the plot in terms of dpi (minimum=200) [Default: None] ')

        group_type.add_option('-L', '--lines_only',\
                              action="store_true",\
                              dest='plot_only_lines',\
                              default=False,\
                              help='Key for plotting lines only (no filling - this overwrites all "fill"-related options) [Default: deactivated] ')

        group_misc.add_option('-i', '--input-system',\
                              action="store",\
                              dest='plot_input_system',\
                              metavar='<basis>',\
                              default=False,\
                              help='Define the coordinate system of the source mechanism - value: NED,USE,XYZ,NWU [Default: NED]  ')

        group_type.add_option('-I', '--show_isotropic_part',\
                              dest='plot_isotropic_part',\
                              action='store_true',\
                              default=False,\
                              help='Key for considering the isotropic part for plotting [Default: deactivated]')


        parser_plot.add_option_group(group_save)
        parser_plot.add_option_group(group_type)
        parser_plot.add_option_group(group_quality)
        parser_plot.add_option_group(group_colours)
        parser_plot.add_option_group(group_misc)
        parser_plot.add_option_group(group_dc)
        parser_plot.add_option_group(group_geo)
        parser_plot.add_option_group(group_app)

        _do_parsers['plot']      = parser_plot



        desc_decomp  = """
Decompose moment tensor into additive contributions.

This method implements four different decompositions following the conventions
given by Jost & Herrmann (1998), and Dahm (1997). The type of decomposition can
be selected with the '--type' option. Use the '--partial' option, if only parts of the full decomposition are required.

By default, the decomposition results are printed in the following order:

    * 01 - basis of the provided input     (string)
    * 02 - basis of the representation     (string)
    * 03 - chosen decomposition type      (integer)

    * 04 - full moment tensor              (matrix)

    * 05 - isotropic part                  (matrix)
    * 06 - isotropic percentage             (float)
    * 07 - deviatoric part                 (matrix)
    * 08 - deviatoric percentage            (float)

    * 09 - DC part                         (matrix)
    * 10 - DC percentage                    (float)
    * 11 - DC2 part                        (matrix)
    * 12 - DC2 percentage                   (float)
    * 13 - DC3 part                        (matrix)
    * 14 - DC3 percentage                   (float)

    * 15 - CLVD part                       (matrix)
    * 16 - CLVD percentage                 (matrix)

    * 17 - seismic moment                   (float)
    * 18 - moment magnitude                 (float)

    * 19 - eigenvectors                   (3-array)
    * 20 - eigenvalues                       (list)
    * 21 - p-axis                         (3-array)
    * 22 - neutral axis                   (3-array)
    * 23 - t-axis                         (3-array)
"""

        parser_decompose         = OptionParser(usage="mopad decompose <source-mechanism> [options]", description=desc_decomp, formatter=MopadHelpFormatter())

        group_type               =  OptionGroup(parser_decompose,'Type of decomposition')
        group_part               =  OptionGroup(parser_decompose,'Output selection')

        group_part.add_option('-p', '--partial',\
                              action="store",\
                              dest='decomp_out_part',\
                              default=None,\
                              metavar='<part1,part2,... >',\
                              help='''
Print a subset of the decomposition results.

Give a comma separated list of what parts of the results should be
printed [Default: None]. The following parts are available:

    %s
''' % ', '.join(decomp_attrib_map_keys))


        group_type.add_option('-t', '--type',\
                              action="store",\
                              dest='decomp_key',\
                              metavar='<decomposition key>',\
                              default=1,\
                              type='int',\
                              help='''
Choose type of decomposition - values 1,2,3,4 \n[Default: 1]:

%s
''' % '\n'.join([ '    * %s - %s' % (k,v[0]) for (k,v) in MomentTensor.decomp_dict.items() ]) )

        parser_decompose.add_option_group(group_type)
        parser_decompose.add_option_group(group_part)
        _add_group_system(parser_decompose)

        _do_parsers['decompose'] = parser_decompose


        parser_describe = OptionParser(
            usage="mopad describe <source-mechanism> [options]",
            description='''
Print the detailed description of a source mechanism


For a given source mechanism, orientations of the fault planes, moment,
magnitude, and moment tensor are printed. Input and output coordinate basis systems
can be specified.
''',

            formatter=MopadHelpFormatter())

        _add_group_system(parser_describe)

        _do_parsers['describe'] = parser_describe

        return _do_parsers




    #------------------------------------------------------------------
    #------------------------------------------------------------------

    if len(sys.argv) < 2:
        call = 'help'

    else:

        call = sys.argv[1].lower()
        abbrev = dict(zip(('p','g','d','i', '--help', '-h'), ('plot','gmt','decompose', 'describe', 'help', 'help')))

        if call in abbrev:
            call = abbrev[call]

        if call not in abbrev.values():
            sys.exit('no such method: %s' % call)


    if call == 'help':
        helpstring = """

Usage: mopad <method> <source-mechanism> [options]


Type 'mopad <method> --help' for help on a specific method.


MoPaD (version %.1f) - Moment Tensor Plotting and Decomposition Tool

MoPaD is a tool to plot and decompose moment tensor representations of seismic
sources which are commonly used in seismology. This tool is completely
controlled via command line parameters, which consist of a <method> and a
<source-mechanism> argument and zero or more options. The <method> argument
tells MoPaD what to do and the <source-mechanism> argument specifies an input
moment tensor source in one of the formats described below.

Available methods:

   * plot:       plot a beachball representation of a source mechanism
   * describe:   print detailed description of a source mechanism
   * decompose:  decompose a source mechanism according to various conventions
   * gmt:        output beachball representation in a format suitable for
                 plotting with GMT

The source-mechanism is given as a comma separated list (NO BLANK SPACES!) of values
which is interpreted differently, according to the number of values in the list. The
following source-mechanism representations are available:

   * strike,dip,rake
   * strike,dip,rake,moment
   * M11,M22,M33,M12,M13,M23
   * M11,M22,M33,M12,M13,M23,moment
   * M11,M12,M13,M21,M22,M23,M31,M32,M33

Angles are given in degrees, moment tensor components and scalar moment are given
in [Nm] for a coordinate system with axes pointing North, East, and Down by
default.
_______________________________________________________________________________

EXAMPLES
--------

'plot' :
--
To generate the "beachball" representation of a pure normal faulting event with
a strike angle of 0 degrees and a dip of 45 degrees, use either of the following
commands:

  mopad plot 0,45,-90

  mopad plot 0,1,-1,0,0,0


'describe':
--
To see the seismic moment tensor entries (in GlobalCMT's USE basis) and the
orientation of the auxilliary plane for a shear crack with the (strike,dip,slip-rake)
tuple (90,45,45) use:

  mopad describe 90,45,45 -o USE


'decompose':
--
Get the deviatoric part of a seismic moment tensor M=(1,2,3,4,5,6) together with
the respective double-couple- and CLVD-components by using:

  mopad decompose 1,2,3,4,5,6 -p devi,dc,clvd


"""%(mopad_version)

        print helpstring

        sys.exit()

    try:
        M_raw = [float(xx) for xx in sys.argv[2].split(',')]
    except:
        dummy_list =  []
        dummy_list.append(sys.argv[0] )
        dummy_list.append(sys.argv[1] )
        dummy_list.append('0,0,0')
        dummy_list.append('-h')

        sys.argv = dummy_list
        M_raw = [float(xx) for xx in sys.argv[2].split(',')]

    if not len(M_raw) in [3,4,6,7,9]:
        print '\nERROR!! Provide proper source mechanism\n\n'
        sys.exit()
    if len(M_raw) in [4,6,7,9] and  len(N.array(M_raw).nonzero()[0]) == 0:
        print '\nERROR!! Provide proper source mechanism\n\n'
        sys.exit()


    aa = _handle_input(call, M_raw, sys.argv[3:],_build_optparsers()[call])
    if aa != None:
        print aa
#------------------------------------------------------------------
# finished
