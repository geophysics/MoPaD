# -*- coding: utf-8 -*-

import mopad_util as util
import mopad_decomposition as decomposition

from mopad_util import die, NED2USE, NED2XYZ, NED2NWU, strikediprake_2_moments,\
                       fancy_matrix, fancy_vector, transform
                       


import numpy as N

pi = N.pi
rad2deg = 180./pi
epsilon = util.epsilon

class MomentTensor:
    
    _m_unrot = N.matrix( [[0.,0.,-1.],[0.,0.,0.],[-1.,0.,0.]], dtype=N.float )
    
    def __init__(self, M=None, in_system='NED', out_system='NED', decomposition_key='standard'):
        """
        Creates a moment tensor object on the basis of a provided mechanism M.

        If M is a non symmetric 3x3-matrix, the upper right triangle
        of the matrix is taken as reference. M is symmetrisised
        w.r.t. these entries. If M is provided as a 3-,4-,6-,7-tuple
        or array, it is converted into a matrix internally according
        to standard conventions (Aki & Richards).

           'in_system' may be one of 'NED','USE','NWU', or 'XYZ'.
           'out_system' may be one of 'NED','USE','NWU', or 'XYZ'.

        """


        source_mechanism       = M
        self._original_M       = M[:]

        self._input_basis = in_system.upper()
        self._output_basis = out_system.upper()

        # bring M to symmetric matrix form
        self._M                 = self._setup_M(source_mechanism, self._input_basis)

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
                      
        #carry out the MT decomposition - results are in basis NED
        self._decomp = decomposition.Decomposition(self._M)
        
        #set the appropriate principal axis system:
        self._M_to_principal_axis_system()


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
            strike, dip, rake = mech
            scalar_moment = 1.0
            if len(mech) == 4:
                scalar_moment = mech[3]
                
            rotmat1 = euler_to_matrix( dip, strike, -rake )
            new_M = rotmat1.T * MomentTensor._m_unrot * rotmat1 * scalar_moment
            
            #to assure right basis system - others are meaningless, provided these angles
            input_basis   =   'NED'
        
        return  transform(N.matrix(new_M), input_basis, 'NED')
            
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
        Helps to construct the R^3.

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


        #print '\nEWs: ', [EW1,EW2,EW3], ' Trace: ',sum([EW1,EW2,EW3]) 
        #print 'EWs devi',[EW1_devi,EW2_devi,EW3_devi], 'trace M devi', trace_M_devi

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
            #if abs(trace_M_devi)< epsilon:
            #    if  abs(EW1_devi) < abs(EW3_devi):
            #symmetry_around_tension = 1
            #clr = 1

#             if EW2_devi < 0:
#                 symmetry_around_tension = 1
#                 clr = 1
              

#             if EW1_devi < 0 and EW2_devi < 0 and  EW3_devi > 0: 
#                 symmetry_around_tension = 1
#                 clr = 1

        else:
            pass

#         #pure deviatoric movement 
#         if trace_M == 0 and  EW[2] == abs(EW[0]):
#             print 'shear'
#             print EW
#             #print EW1
#             #exit()
            
#             if EW[2] != abs(EW[0]):
#                 print 'CLVD'

#                 if EW[2] < abs(EW[0]):
#                     print 'symmetry around tension ( red)'
                                   
#                     symmetry_around_tension = 0
#                     clr = 1
#                 else:
#                     print 'symmetry around pressure (white)'
                                   
#                     symmetry_around_tension = 0
#                     clr = -1
                    
                
#                 if EW[2] > abs( EW[0]):
#                     symmetry_around_tension = 0
                
#                     clr = -1
                    
            
#             # elif abs(EW3) == EW1 and EW[0] > EW[2]: 
# #                 symmetry_around_tension = 0
# #                 #logger.debug( 'SIGMA AXIS = tension\n')
# #                 clr = 1

#             else:
#                 symmetry_around_tension = 1
#                 #logger.debug( 'SIGMA AXIS = tension\n')
#                 clr = -1
            

# #         elif trace_M == 0:
# #             symmetry_around_tension = 1
# #             if 
                            
# #             print 'detmat', det_mat
# #             if det_mat > 0:
# #                 symmetry_around_tension = 0
# #                 clr = 1
# #                 pass        

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
            raise MTError('check M: ( Trace(M) > 0, but largest eigenvalue is still negative)')


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
            
                
         
        #print 'symm around tension:  ',symmetry_around_tension
        #exit()
        #symmetry_around_tension = 0
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
    
        # print 'HNS: ',[EWh,EWn,EWs] 
#         print 
#         print '123: ', [EW1,EW2,EW3 ]
#         print 
#         print 'colour order', clr
#         print
        #exit()
    
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

    

    def _find_faultplanes(self):
        
        """
        Sets the two angle-triples, describing the faultplanes of the
        Double Couple, defined by the eigenvectors P and T of the
        moment tensor object.
        
        
        Defining a reference Double Couple with strike = dip =
        slip-rake = 0, the moment tensor object's DC is transformed
        (rotated) w.r.t. this orientation. The respective rotation
        matrix yields the first fault plane angles as the Euler
        angles. After flipping the first reference plane by
        multiplying the appropriate flip-matrix, one gets the second fault
        plane's geometry.

        All output angles are in degree

        (
        to check:
        mit Sebastians Konventionen:

        rotationsmatrix1 = EV Matrix von M, allerdings in der Reihenfolge TNP (nicht, wie hier PNT!!!)

        referenz-DC mit strike, dip, rake = 0,0,0  in NED - Darstellung:  M = 0,0,0,0,-1,0

        davon die EV ebenfalls in eine Matrix:

        trafo-matrix2 = EV Matrix von Referenz-DC in der REihenfolge TNP

        effektive Rotationsmatrix = (rotationsmatrix1  * trafo-matrix2.T).T

        durch check, ob det <0, schauen, ob die Matrix mit -1 multipliziert werden muss

        flip_matrix = 0,0,-1,0,-1,0,-1,0,0

        andere DC Orientierung wird durch flip * effektive Rotationsmatrix erhalten

        beide Rotataionmatrizen in matrix_2_euler
        )

        """

        # reference Double Couple (in NED basis) - it has strike, dip, slip-rake = 0,0,0
        refDC                    = N.matrix( [[0.,0.,-1.],[0.,0.,0.],[-1.,0.,0.]], dtype=N.float )
        refDC_evals, refDC_evecs = N.linalg.eigh(refDC)

        #matrix which is turning from one fault plane to the other
        flip_dc                  = N.matrix( [[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]], dtype=N.float )

        #euler-tools need matrices of EV sorted in PNT:
        pnt_sorted_EV_matrix      = self._rotation_matrix.copy() 

        #resort only necessary, if abs(p) <= abs(t)
        #print self._plot_clr_order
        if self._plot_clr_order < 0:
            pnt_sorted_EV_matrix[:,0] = self._rotation_matrix[:,2]
            pnt_sorted_EV_matrix[:,2] = self._rotation_matrix[:,0]

        # rotation matrix, describing the rotation of the eigenvector
        # system of the input moment tensor into the eigenvector
        # system of the reference Double Couple
        rot_matrix_fp1       = (N.dot(pnt_sorted_EV_matrix, refDC_evecs.T)).T

        #check, if rotation has right orientation
        if N.linalg.det(rot_matrix_fp1) < 0.:
            rot_matrix_fp1       *= -1.

        #adding a rotation into the ambiguous system of the second fault plane
        rot_matrix_fp2       = N.dot(flip_dc,rot_matrix_fp1)

        fp1                  = self._find_strike_dip_rake(rot_matrix_fp1)
        fp2                  = self._find_strike_dip_rake(rot_matrix_fp2)
    
        return  [fp1,fp2]
    
    
    def _find_strike_dip_rake(self,rotation_matrix):

        """
        Returns angles strike, dip, slip-rake in degrees, describing the fault plane.

        """

        (alpha, beta, gamma) = self._matrix_to_euler(rotation_matrix)
       
    
        return (beta*rad2deg, alpha*rad2deg, -gamma*rad2deg)
    
    



    def _matrix_w_system(self, M2return):
        """
        Gives the provided matrix in the desired basis system.
        """
        
     
        return transform(self, M2return, 'NED', self._output_basis)

    

    def _vector_w_system(self, vectors, system):
        """
        Gives the provided vector(s) in the desired basis system.

        'vectors' can be either a single array, tuple, matrix or a collection in form of a list, array or matrix.
        If it's a list, each entry will be checked, if it's 3D - if not, an exception is raised.
        If it's a matrix or array with column-length 3, the columns are interpreted as vectors, otherwise, its transposed is used.

        """
        
        if not system.upper() in self._list_of_possible_output_bases:
            raise MTError('provided output basis not supported - please specify'+
                          ' one of the following bases: %s (default=NED)\n' 
                            % ', '.join( self._list_of_possible_input_bases ))
        
        
        lo_vectors = []

        # if list of vectors
        if type(vectors) == list:
            for vec in vectors:
                if N.prod(N.shape(vec)) != 3:
                    raise MTError('please provide vector(s) from R³')
                
            lo_vectors = vectors
        
        else:
            if N.prod(N.shape(vectors))%3 != 0:
                raise MTError('please provide vector(s) from R³')

            if N.shape(vectors)[0] == 3:
                for ii in  N.arange(N.shape(vectors)[1]) :
                    lo_vectors.append(vectors[:,ii])
            else: 
                for ii in  N.arange(N.shape(vectors)[0]) :
                    lo_vectors.append(vectors[:,ii].transpose())

        lo_vecs_to_show = []
        
        for vec in lo_vectors:

            if system.upper() == 'NED':
                lo_vecs_to_show.append(vec)

            elif system.upper() == 'USE':
                lo_vecs_to_show.append( NED2USE(vec))

            elif system.upper() == 'XYZ':
                lo_vecs_to_show.append( NED2XYZ(vec))

            elif system.upper() == 'NWU':
                lo_vecs_to_show.append( NED2NWU(vec))

        if len(lo_vecs_to_show) == 1 :
            return lo_vecs_to_show[0]
        else:
            return lo_vecs_to_show

    

    def get_M(self,system='NED'):
        """
        Returns the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy style (to be viewed with 'print')
        """
        
        return  self._matrix_w_style_and_system(self._M,system)


    def get_decomposition(self,in_system='NED',out_system='NED'):
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
        -24 - faultplanes       (list of two 3-arrays)
  
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
    

    def get_full_decomposition(self):
        """
        Nice compilation of decomposition result to be viewed in the shell (call with 'print').
        """

        mexp = pow(10,N.ceil(N.log10(N.max(N.abs(self._M)))))
        m =  self._M/mexp
        s =  '\nScalar Moment: M0 = %g Nm (Mw = %3.1f)\n' 
        s += 'Moment Tensor: Mnn = %6.3f,  Mee = %6.3f, Mdd = %6.3f,\n'
        s += '               Mne = %6.3f,  Mnd = %6.3f, Med = %6.3f    [ x %g ]\n\n'
        s = s % (self._seismic_moment, self._moment_magnitude, m[0,0],m[1,1],m[2,2],m[0,1],m[0,2],m[1,2], mexp)
        
        s += self._fault_planes_as_str()
        return s
    

    def _fault_planes_as_str(self):
        """
        Internal setup of a nice string, containing information about the fault planes.
        """
        s = '\n'
        for i,sdr in enumerate(self.get_fps()):
            s += 'Fault plane %i: strike = %3.0f°, dip = %3.0f°, slip-rake = %4.0f°\n' % \
                 (i+1, sdr[0], sdr[1], sdr[2])           
        return s

    
    def get_input_system(self):
        """
        Returns the basis system of the input.
        """
        
        return  self._input_basis
    
    
    def get_eigvals(self, system='NED'):
        """
        Returns a list of the eigenvalues of the moment tensor.
        """
        
        # in the order HNS:
        return self._eigenvalues

    
    def get_eigvecs(self, system='NED'):
        """
        Returns the eigenvectors  of the moment tensor.

        Call with arguments to set ouput in other basis system
        """

        return self._vector_w_style_and_system(self._eigenvectors, system)
        
    
    def get_null_axis(self, system='NED'):
        """
        Returns the neutral axis of the moment tensor.

        Call with arguments to set ouput in other basis system
        """

        return self._vector_w_style_and_system(self._null_axis, system)

    
    def get_t_axis(self, system='NED'):
        """
        Returns the tension axis of the moment tensor.

        Call with arguments to set ouput in other basis system
        """
         
        return self._vector_w_style_and_system(self._t_axis, system)
 
    
    def get_p_axis(self, system='NED'):
        """
        Returns the pressure axis of the moment tensor.

        Call with arguments to set ouput in other basis system
        """
        
        return self._vector_w_style_and_system(self._p_axis, system)

    
    def get_transform_matrix(self, system='NED'):
        """
        Returns the  transformation matrix (input system to principal axis system.

        Call with arguments to set ouput in other basis system
        """
        
        return  self._matrix_w_style_and_system(self._rotation_matrix, system)


    def get_fps(self):
        """
        Returns a list of the two faultplane 3-tuples, each showing strike, dip, slip-rake.
        """

        return self._faultplanes
    
    
    def get_colour_order(self):
        """
        Returns the value of the plotting order (only important in BeachBall instances).
        """
        
        return self._plot_clr_order


    @staticmethod
    def validate_projection(projection):
        
        projection = projection.lower()
        projection_abbr = dict(zip(('s','o','l','g'), ('stereo','ortho','lambert','gnom')))
        
        if projection in projection_abbr:
            projection = projection_abbr[projection]
            
        projections = MomentTensor.projections
        if not self._plot_projection in projections:
            raise MTError('projection %s not available - choose from: %s' % 
			  (self._plot_projection, ', '.join( projections.keys() ) ) 
			  )

        return projection

print MomentTensor.__dict__
exit()

MomentTensor.decomp_dict = {
    '1': ('ISO + DC + CLVD',                 decomposition.Decomposition(MomentTensor._) ),
    '2': ('ISO + major DC + minor DC',       MomentTensor._decomposition_w_2DC),
    '3': ('ISO + DC1 + DC2 + DC3',           MomentTensor._decomposition_w_3DC),
    '4': ('ISO + strike DC + dip DC + CLVD', MomentTensor._decomposition_w_CLVD_2DC),
}

MomentTensor.projections = {
    'stereo':   MomentTensor._stereo_vertical,
    'ortho':    MomentTensor._orthographic_vertical,
    'lambert':  MomentTensor._lambert_vertical,
    'gnom':     MomentTensor._gnomonic_vertical,
}

MomentTensor.decomp_part_methods = dict([
    (k, getattr(MomentTensor, 'get_'+v)) for (k,v) in 
        zip( ('in','out','type',\
            'full','m',\
            'iso','iso_perc',\
            'dev','devi','devi_perc',\
            'dc','dc_perc',\
            'dc2','dc2_perc',\
            'dc3','dc3_perc',\
            'clvd','clvd_perc',\
            'mom','mag',\
            'eigvals','eigvecs',\
            't','n','p',\
            'fps','faultplanes','fp',\
            'decomp_key'),
            ('input_system','output_system','decomp_type',\
            'M','M',\
            'iso','iso_percentage',\
            'devi','devi','devi_percentage',\
            'DC','DC_percentage',\
            'DC2','DC2_percentage',\
            'DC3','DC3_percentage',\
            'CLVD','CLVD_percentage',\
            'moment','mag',\
            'eigvals','eigvecs',\
            't_axis','null_axis','p_axis',\
            'fps','fps','fps',\
            'decomp_type') ) ] ) 
