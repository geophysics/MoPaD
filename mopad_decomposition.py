import numpy as N
from mopad_util import epsilon, fancy_matrix

class Decomposition:
    """
    Standard decomposition according to Aki & Richards and Jost & Herrmann into

    - isotropic
    - deviatoric
    - DC
    - CLVD

    parts of the input moment tensor.

    DC, CLVD, DC_percentage, seismic_moment, moment_magnitude

    """

    def __init__(self, M):
        self._M = M

        self._seismic_moment        = None
        self._seismic_moment_jost_herrmann = None
        self._decompose()

    @staticmethod
    def part_names():
        return ['Isotropic', 'Deviatoric', 'DC', 'CLVD']

    @staticmethod
    def decomposition_info():
        return 'Isotropic + Deviatoric = Isotropic + (DC + CLVD)'

    def parts(self):
        return self._parts

    def percentages(self):
        return self._percentages

    def part_id(self, name):
        try:
            i = [n.lower() for n in self.part_names()].index(name.lower())
        except ValueError:
            raise InvalidDecompositionPart(name)

        return i

    def part(self, name):
        return self._parts[self.part_id(name)]

    def percentage(self, name):
        return self._percentages[self.part_id(name)]

    def moment(self):
        return self._seismic_moment

    def magnitude(self):
        return N.log10(self._seismic_moment*1.0e7)/1.5 - 10.7
    
    def __getattr__(self, name):
        def getter():
            if name.endswith('_percentage'):
                return self.percentage(name[:-len('_percentage')])
            else:
                return self.part(name)
        
        return getter
        
    def dict(self):
        d = {}
        for name in self.part_names():
            lname = name.lower()
            d[lname] = self.part(name)
            d[lname+'_fancy'] = fancy_matrix(self.part(name))
            d[lname+'_percentage'] = self.percentage(name)
        
        d['moment'] = self.moment()
        d['magnitude'] = self.magnitude()
        
        return d
        
    def fill(self, s):
        return s % self.dict()
    
    def __str__(self):
        d = self.dict()
        ks = ['Moment magnitude', 'Seismic moment']
        vs = ['%(magnitude)3.1f', '%(moment)g Nm']
        
        for n in self.part_names():
            ks.append('%s percentage' % n)
            vs.append('%%(%s_percentage)3.0f %%%%' % n.lower())
        
        for n in self.part_names():
            ks.append('%s' % n)
            vs.append('%%(%s_fancy)s' % n.lower())
            
        kl = max([len(k) for k in ks])
        return '\n'.join([('%-'+str(kl+1)+'s %s') % ( k+':', v % d) for k, v in zip(ks,vs) ])
        
    def _decompose(self):

        M = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso


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

        M_DC        = eigenw[2]*(1-2*F)*( N.outer(a3,a3) - N.outer(a2,a2) )
        M_CLVD      = M_devi - M_DC #eigenw[2]*F*( 2*N.outer(a3,a3) - N.outer(a2,a2) - N.outer(a1,a1))

        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi

        M_iso_percentage = M0_iso/M0 *100
        M_DC_percentage = ( 1 - 2 * abs(F) )* ( 1 - M_iso_percentage/100.)  * 100

        self._parts = M_iso, M_devi, M_DC, M_CLVD
        self._percentages = (M_iso_percentage, 100. - M_iso_percentage,
            M_DC_percentage,
            100. - M_DC_percentage - M_iso_percentage )

        self._seismic_moment    = M0
        self._seismic_moment_jost_herrmann = N.sqrt(1./2*N.sum(eigenw**2) )


class Decomposition_2DC(Decomposition):
    """
    Decomposition according Aki & Richards and Jost & Herrmann into

    - isotropic
    - deviatoric
    - 2 DC

    parts of the input moment tensor.

    results are given as attributes, callable via the get_* function:

    DC1, DC2, DC_percentage, seismic_moment, moment_magnitude

    """

    @staticmethod
    def part_names():
        return ['Isotropic', 'Deviatoric', 'DC_major', 'DC2_minor']

    @staticmethod
    def decomposition_info():
        return 'Isotropic + Deviatoric = Isotropic + (DC_major + DC_minor)'

    def _decompose(self):
        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso


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

        M_DC        = eigenw[2]*( N.outer(a3,a3) - N.outer(a2,a2) )
        M_DC2       = eigenw[0]*( N.outer(a1,a1) - N.outer(a2,a2) )

        M_DC_percentage = abs(eigenw[2]/(abs(eigenw[2])+abs(eigenw[0]) )     )

        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi

        M_iso_percentage     = M0_iso/M0 * 100.

        self._parts = M_iso, M_devi, M_DC, M_DC2
        self._percentages = (M_iso_percentage, 100. - M_iso_percentage,
            M_DC_percentage, 100. - M_DC_percentage - M_iso_percentage )

        self._seismic_moment    = M0
        self._seismic_moment_jost_herrmann = N.sqrt(1./2*N.sum(eigenw**2) )


class Decomposition_CLVD_2DC(Decomposition):
    """
    Decomposition according to Dahm (1993) into

    - isotropic
    - CLVD
    - strike-slip
    - dip-slip

    parts of the input moment tensor.

    results are given as attributes, callable via the get_* function:

    iso, CLVD, DC1, DC2, iso_percentage, DC_percentage, DC1_percentage, DC2_percentage, CLVD_percentage, seismic_moment, moment_magnitude

    """

    @staticmethod
    def part_names():
        return ['Isotropic', 'Deviatoric', 'CLVD', 'DC_strike', 'DC_dip']

    @staticmethod
    def decomposition_info():
        return 'Isotropic + Deviatoric = Isotropic + (CLVD + DC_strike + DC_dip)'

    def _decompose(self):

        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

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

        M_iso_percentage     = M0_M_iso/M0_M * 100.
        M_DC_percentage      = M0_M_dc/M0_M * 100.
        M_DC1_percentage     = M0_M_dc1/M0_M * 100.
        M_DC2_percentage     = M0_M_dc2/M0_M * 100.

        self._parts = M_iso, M_devi, M_CLVD, M_DC1, M_DC2
        self._percentages = (M_iso_percentage, 100. - M_iso_percentage,
            100. - M_DC1_percentage - M_DC2_percentage - M_iso_percentage,
            M_DC1_percentage, M_DC2_percentage)

        self._seismic_moment    = M0_M
        self._seismic_moment_jost_herrmann   = N.sqrt(1./2*N.sum(eigvals_M**2) )


class Decomposition_3DC(Decomposition):

    """
    Decomposition according Aki & Richards and Jost & Herrmann into

    - isotropic
    - deviatoric
    - 3 DC

    parts of the input moment tensor.

    results are given as attributes, callable via the get_* function:

    DC1, DC2, DC3, DC_percentage, seismic_moment, moment_magnitude

    """

    @staticmethod
    def part_names():
        return ['Isotropic', 'Deviatoric', 'DC1', 'DC2', 'DC3']

    @staticmethod
    def decomposition_info():
        return 'Isotropic + Deviatoric = Isotropic + (DC1 + DC2 + DC3)'

    def _decompose(self):

        M      = self._M

        #isotropic part
        M_iso   = N.diag( N.array([1./3*N.trace(M),1./3*N.trace(M),1./3*N.trace(M)] ) )
        M0_iso  = abs(1./3*N.trace(M))

        #deviatoric part
        M_devi  = M - M_iso

        #eigenvalues and -vectors of the deviatoric part
        eigenw1,eigenv1  = N.linalg.eig(M_devi)
        M0_devi          = max(abs(eigenw1))

        #eigenvalues and -vectors of the full M !!!!!!!!
        eigenw1,eigenv1  = N.linalg.eig(M)


        #eigenvalues in ascending order of their absolute values:
        eigenw           = N.real( N.take( eigenw1,N.argsort(abs(eigenw1)) ) )
        eigenv           = N.real( N.take( eigenv1,N.argsort(abs(eigenw1)), 1 ) )


        #named according to Jost & Herrmann:
        a1 = eigenv[:,0]
        a2 = eigenv[:,1]
        a3 = eigenv[:,2]

        M_DC1        = 1./3.*(eigenw[0] - eigenw[1]) *( N.outer(a1,a1) - N.outer(a2,a2) )
        M_DC2        = 1./3.*(eigenw[1] - eigenw[2]) *( N.outer(a2,a2) - N.outer(a3,a3) )
        M_DC3        = 1./3.*(eigenw[2] - eigenw[0]) *( N.outer(a3,a3) - N.outer(a1,a1) )

        M_DC1_perc = 100.*abs((eigenw[0]-eigenw[1])) / (abs((eigenw[1]-eigenw[2]))+abs((eigenw[1]-eigenw[2]))+abs((eigenw[2]-eigenw[0])))
        M_DC2_perc = 100.*abs((eigenw[1]-eigenw[2])) / (abs((eigenw[1]-eigenw[2]))+abs((eigenw[1]-eigenw[2]))+abs((eigenw[2]-eigenw[0])))

        #according to Bowers & Hudson:
        M0          = M0_iso + M0_devi
        M_iso_percentage     = M0_iso/M0 *100.

        self._parts = M_iso, M_devi, M_DC1, M_DC2, M_DC3
        self._percentages = (M_iso_percentage, 100. - M_iso_percentage,
            M_DC1_percentage, M_DC2_percentage,
            100. - M_DC1_percentage - M_DC2_percentage - M_iso_percentage )

        self._seismic_moment    = M0
        self._seismic_moment_jost_herrmann = N.sqrt(1./2*N.sum(eigenw**2) )
