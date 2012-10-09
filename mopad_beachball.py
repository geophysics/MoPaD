
from mopad_util import epsilon

import numpy as N
from cStringIO import StringIO

pi = N.pi
rad2deg = 180./pi

class BeachBall:
    """
    Class for generating a beachball projection for a provided moment tensor object.


    Input: a MomentTensor object

    Output can be plots of
    - the eigensystem
    - the complete sphere
    - the projection to a unit sphere
      .. either lower (standard) or upper half

    Beside the plots, the unit sphere projection may be saved in a given file.

    Alternatively, only the file can be provided without showing anything directly.
    """

    def __init__(self, MT=MomentTensor, kwargs_dict={}):

        self.MT = MT
        self._M = MT._M
        self._set_standard_attributes()
        self._update_attributes(kwargs_dict)
        self._nodallines_in_NED_system()
        
        #self._identify_faultplanes()

    
    #-------------------------------------------------------------------

    def ploBB(self, kwargs):
        """
        Plots the projection of the beachball onto a unit sphere.

        Module matplotlib (pylab) must be installed !!! 
        """

        self._update_attributes(kwargs)

        self._setup_BB()

        self._plot_US()
 
    #-------------------------------------------------------------------

    def save_BB(self, kwargs):
        """
        Saves the 2D projection of the beachball without plotting.

        Module matplotlib (pylab) must be installed !!! 
       
        keyword arguments:

        - outfile : name of outfile, addressing w.r.t. current directory
        - format  : if no implicit valid format is provided within the filename, add file format        
        """

        self._update_attributes(kwargs)

        self._setup_BB()

        self._just_save_bb()
        
    #-------------------------------------------------------------------

    def _just_save_bb(self):
        """
        Saves the beachball unit sphere plot into a given  file.
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
                #matplotlib.use('GDK') 
            
            
                
#             try:
#                 matplotlib.use('GTKCairo')
                
#             except:
#                 matplotlib.use('GTKCairo')
#                 pass


        import pylab as P
        #        import os.path as op
        #        import os
        
        plotfig = self._setup_plot_US(P)

        outfile_format = self._plot_outfile_format
        outfile_name   = self._plot_outfile

        outfile_abs_name = op.realpath(op.abspath(op.join(os.curdir, outfile_name)))

        try:
            plotfig.savefig(outfile_abs_name, dpi=self._plot_dpi, transparent=True, format=outfile_format)
            
        except:
            raise MTError('Saving of plot failed')

        P.close(667)

        del P
        del matplotlib

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    
    def get_psxy(self, kwargs):
        """
        Returns one string, to be piped into psxy of GMT.

        keyword arguments and defaults:
        
        - GMT_type         = fill/lines/EVs (select type of string - default = fill)
        - GMT_scaling      = 1.             (scale the beachball - original radius is 1)
        - GMT_tension_colour  = 1              (tension area of BB -- colour flag for -Z in psxy)
        - GMT_pressure_colour= 0              (pressure area of BB -- colour flag for -Z in psxy) 
        - GMT_show_2FPs    = 0              (flag, if both faultplanes are to be shown)
        - GMT_show_1FP     = 1              (flag, if one faultplane is to be shown)   
        - GMT_FP_index     = 2              (which one -- 1 or 2 )

        """
 
        self._GMT_type          = 'fill'
        self._GMT_2fps          = False
        self._GMT_1fp           = 0

        self._GMT_psxy_fill     = None
        self._GMT_psxy_nodals  = None
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

    def _add_2_GMT_string(self, FH_string, curve, colour):
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
            self._add_2_GMT_string(GMT_string_FH, US, pressure_colour)
            self._add_2_GMT_string(GMT_string_FH, neg_nodalline, tension_colour)
            self._add_2_GMT_string(GMT_string_FH, pos_nodalline, tension_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH, US, tension_colour)

                if self._plot_curve_in_curve < 1 :
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline, pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline, tension_colour)

                    GMT_string_FH.flush()
                
                else:
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline, pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline, tension_colour)

                    GMT_string_FH.flush()

        else:
            self._add_2_GMT_string(GMT_string_FH, US, tension_colour)
            self._add_2_GMT_string(GMT_string_FH, neg_nodalline, pressure_colour)
            self._add_2_GMT_string(GMT_string_FH, pos_nodalline, pressure_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH, US, pressure_colour)
                
                if self._plot_curve_in_curve < 1 :
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline, tension_colour)
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline, pressure_colour)

                    GMT_string_FH.flush()
                
                else:
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline, tension_colour)
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline, pressure_colour)

                    GMT_string_FH.flush()

        # set all nodallines and faultplanes for plotting:
        #
        self._add_2_GMT_string(GMT_linestring_FH, neg_nodalline, tension_colour)
        self._add_2_GMT_string(GMT_linestring_FH, pos_nodalline, tension_colour)


        if self._GMT_2fps :
            self._add_2_GMT_string(GMT_linestring_FH, FP1_2_plot, tension_colour)
            self._add_2_GMT_string(GMT_linestring_FH, FP2_2_plot, tension_colour)

        elif self._GMT_1fp:
            if int(self._GMT_1fp) == 1:
                self._add_2_GMT_string(GMT_linestring_FH, FP1_2_plot, tension_colour)
            elif int(self._GMT_1fp) == 2:
                self._add_2_GMT_string(GMT_linestring_FH, FP2_2_plot, tension_colour)
        
        self._add_2_GMT_string(GMT_linestring_FH, US, tension_colour)

        GMT_linestring_FH.flush()

        setattr(self, '_GMT_psxy_nodals', GMT_linestring_FH)
        setattr(self, '_GMT_psxy_fill', GMT_string_FH)
        setattr(self, '_GMT_psxy_EVs', GMT_EVs_FH)

    #-------------------------------------------------------------------
    def get_MT(self):
        """
        Returns the original moment tensor object, handed over to the class at generating this instance.
        """
        return self.MT

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def full_sphere_plot(self, kwargs): 
        """
        Plot of the full beachball, projected on a circle with a radius 2.
        
        Module matplotlib (pylab) must be installed !!!


        keyword arguments:
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
        plotfig = P.figure(665, figsize=(self._plot_aux_plot_size, self._plot_aux_plot_size) )

        plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax = plotfig.add_subplot(111, aspect='equal')
        #P.axis([-1.1,1.1,-1.1,1.1],'equal')
        ax.axison = False
    
        EV_2_plot        = getattr(self, '_all_EV'+'_final')
        BV_2_plot        = getattr(self, '_all_BV'+'_final').transpose()
        curve_pos_2_plot = getattr(self, '_nodalline_positive'+'_final')
        curve_neg_2_plot = getattr(self, '_nodalline_negative'+'_final')
        FP1_2_plot       = getattr(self, '_FP1'+'_final')
        FP2_2_plot       = getattr(self, '_FP2'+'_final')

        tension_colour      =  self._plot_tension_colour
        pressure_colour    =  self._plot_pressure_colour

        
            
        if self._plot_clr_order > 0 :
            if self._plot_fill_flag:

                ax.fill( self._outer_circle[0,:],self._outer_circle[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                
        
                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0,:], self._outer_circle[1,:],fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                    if self._plot_curve_in_curve < 1:
                        ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        
        
                    else:
                        ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        
            if self._plot_show_princ_axes:
      
                ax.plot( [EV_2_plot[0,0]], [EV_2_plot[1,0]], 'm^', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,3]], [EV_2_plot[1,3]], 'mv', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,1]], [EV_2_plot[1,1]], 'b^', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,4]], [EV_2_plot[1,4]], 'bv', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,2]], [EV_2_plot[1,2]], 'g^', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)
                ax.plot( [EV_2_plot[0,5]], [EV_2_plot[1,5]], 'gv', ms=self._plot_princ_axes_symsize , lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)        
                
        else:
            if self._plot_fill_flag:
                ax.fill( self._outer_circle[0,:], self._outer_circle[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
        
                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0,:], self._outer_circle[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha )
                    if self._plot_curve_in_curve < 0 :
                        ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass
                    else:
                        ax.fill( curve_pos_2_plot[0,:], curve_pos_2_plot[1,:], fc=tension_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        ax.fill( curve_neg_2_plot[0,:], curve_neg_2_plot[1,:], fc=pressure_colour, alpha= self._plot_fill_alpha*self._plot_total_alpha)
                        pass
        
            if self._plot_show_princ_axes:

                ax.plot( [EV_2_plot[0,0]], [EV_2_plot[1,0]], 'g^', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha) 
                ax.plot( [EV_2_plot[0,3]], [EV_2_plot[1,3]], 'gv', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha) 
                ax.plot( [EV_2_plot[0,1]], [EV_2_plot[1,1]], 'b^', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha) 
                ax.plot( [EV_2_plot[0,4]], [EV_2_plot[1,4]], 'bv', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha) 
                ax.plot( [EV_2_plot[0,2]], [EV_2_plot[1,2]], 'm^', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha) 
                ax.plot( [EV_2_plot[0,5]], [EV_2_plot[1,5]], 'mv', ms=self._plot_princ_axes_symsize, lw=self._plot_princ_axes_lw , alpha=self._plot_princ_axes_alpha*self._plot_total_alpha)    
        
        self._plot_nodalline_colour='y'
        
        ax.plot( curve_neg_2_plot[0,:] , curve_neg_2_plot[1,:], 'o', c=self._plot_nodalline_colour, lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha , ms=3 )

        self._plot_nodalline_colour='b'

        ax.plot( curve_pos_2_plot[0,:] , curve_pos_2_plot[1,:], 'D', c=self._plot_nodalline_colour, lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha , ms=3)

        if self._plot_show_1faultplane:
            if self._plot_show_FP_index == 1:
                ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:], '+', c=self._plot_faultplane_colour, lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha, ms=5)

            elif self._plot_show_FP_index == 2:
                ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:], '+', c=self._plot_faultplane_colour, lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha, ms=5)

        elif self._plot_show_faultplanes :
            ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:], '+', c=self._plot_faultplane_colour, lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha, ms=4)
            ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:], '+', c=self._plot_faultplane_colour, lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha, ms=4)
    
        else:
            pass

        #if isotropic part shall be displayed, fill the circle completely with the appropriate colour
        if self._pure_isotropic:
            if abs( N.trace( self._M )) > epsilon:
                if self._plot_clr_order < 0:
                    ax.fill(self._outer_circle[0,:], self._outer_circle[1,:], fc=tension_colour, alpha= 1, zorder=100 )
                else:
                    ax.fill( self._outer_circle[0,:], self._outer_circle[1,:], fc=pressure_colour, alpha= 1, zorder=100 )

        #plot NED basis vectors
        if self._plot_show_basis_axes:
            
            plot_size_in_points = self._plot_size * 2.54 * 72
            points_per_unit = plot_size_in_points/2.
            
            fontsize  = plot_size_in_points / 66.
            symsize   = plot_size_in_points / 77.

            direction_letters = list('NSEWDU')
            for idx, val in enumerate(BV_2_plot):
                x_coord = val[0]
                y_coord = val[1]
                np_letter = direction_letters[idx]
            
                rot_angle    = - N.arctan2(y_coord, x_coord) + pi/2.
                original_rho = N.sqrt(x_coord**2 + y_coord**2)
            
                marker_x  = ( original_rho - ( 3* symsize  / points_per_unit ) ) * N.sin( rot_angle )
                marker_y  = ( original_rho - ( 3* symsize  / points_per_unit ) ) * N.cos( rot_angle )
                annot_x   = ( original_rho - ( 8.5* fontsize / points_per_unit ) ) * N.sin( rot_angle )
                annot_y   = ( original_rho - ( 8.5* fontsize / points_per_unit ) ) * N.cos( rot_angle )
            
                ax.text(annot_x, annot_y, np_letter, horizontalalignment='center', size=fontsize, weight='bold', verticalalignment='center',\
                        bbox=dict(edgecolor='white', facecolor='white', alpha=1))
                
                if original_rho > epsilon:
                    ax.scatter([marker_x], [marker_y], marker=(3,0,rot_angle) , s=symsize**2, c='k', facecolor='k', zorder=300)
                else:
                    ax.scatter([x_coord], [y_coord], marker=(4,1,rot_angle) , s=symsize**2, c='k', facecolor='k', zorder=300)



        #plot both circle lines (radius 1 and 2)
        ax.plot(self._unit_sphere[0,:], self._unit_sphere[1,:] , c=self._plot_outerline_colour, lw=self._plot_outerline_width, alpha= self._plot_outerline_alpha*self._plot_total_alpha)# ,ls=':')
        ax.plot(self._outer_circle[0,:], self._outer_circle[1,:], c=self._plot_outerline_colour, lw=self._plot_outerline_width, alpha= self._plot_outerline_alpha*self._plot_total_alpha)# , ls=':')

        #dummy points for setting plot plot size more accurately
        ax.plot([0,2.1,0,-2.1],[2.1,0,-2.1,0],',',alpha=0.)
    
        ax.autoscale_view(tight=True, scalex=True, scaley=True)
        interactive(True)


        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile+'.'+self._plot_outfile_format, dpi=self._plot_dpi, transparent=True, format=self._plot_outfile_format)
            except:
                raise MTError( 'saving of plot failed' )

        P.show()

        del P
        del matplotlib       

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def pa_plot(self, kwargs):
        """
        Plot of the solution in the principal axes system.

        Module matplotlib (pylab) must be installed !!!

        keyword arguments:
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
        
        fig = P.figure(34, figsize=(size, size))
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

                   
        t_angles = N.arange(0., 360., 90)
        t_labels = [' N ', ' H ', ' - N', ' - H']
        
        P.thetagrids( t_angles, labels=t_labels )
            
        ax.plot(self._phi_curve, r_hor   , color='r', lw=3)
        ax.plot(self._phi_curve, r_hor_FP, color='b', lw=1.5)
        ax.set_rmax(1.0)
        P.grid(True)

        P.rgrids((r_steps), labels=r_labels)

        ax.set_title("beachball in eigenvector system", fontsize=15)



        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile+'.'+self._plot_outfile_format, dpi=self._plot_dpi, transparent=True, format=self._plot_outfile_format)
            
            except:
                raise MTError('saving of plot failed')

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
        self._plot_outfile_format    = 'svg'
    
    #---------------------------------------------------------------

    def _update_attributes(self, kwargs):
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
                setattr(self, '_'+kw, kwargs[key])

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

        list_of_curves_2_correct = ['nodalline_negative', 'nodalline_positive', 'FP1', 'FP2']
        projection = self._plot_projection

        n_curve_points = self._plot_n_points

        for obj in list_of_curves_2_correct:
            obj2cor_name = '_'+obj+'_2D'
            obj2cor = getattr(self, obj2cor_name)

            obj2cor_in_right_order = self._sort_curve_points(obj2cor)

            #logger.debug( 'curve: ', str(obj))
            # check, if curve closed !!!!!!
            start_r               = N.sqrt(obj2cor_in_right_order[0,0]**2  + obj2cor_in_right_order[1,0]**2 )
            r_last_point          = N.sqrt(obj2cor_in_right_order[0,-1]**2 + obj2cor_in_right_order[1,-1]**2   )
            dist_last_first_point = N.sqrt( (obj2cor_in_right_order[0,-1] - obj2cor_in_right_order[0,0])**2 + (obj2cor_in_right_order[1,-1] - obj2cor_in_right_order[1,0] )**2 )
            

            # check, if distance between last and first point is smaller than the distance between last point and the edge (at radius=2) 
            if dist_last_first_point > (2 - r_last_point):
                #add points on edge to polygon, if it is an open curve
                #logger.debug( str(obj)+' not closed - closing over edge... ')
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
                #logger.debug( 'open angle %.2f degrees - filling with %i points on the edge\n'%(openangle/pi*180,n_edgepoints))

        
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

            
            setattr(self, '_'+obj+'_in_order', obj2cor_in_right_order)

        return 1



    #---------------------------------------------------------------

    def _nodallines_in_NED_system(self):
        """
        The two nodal lines between the areas on a beachball are given by the points, where
        tan²(alpha) = (-EWs/(EWN*cos(phi)**2 + EWh*sin(phi)**2))
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

#             iso_perc   = self.MT._iso_percentage
#             sign_colour= self._plot_clr_order #positiv, wenn symmetrie um explos. komp.
#             sign_trace = N.sign(N.trace(self._M)) # positiv, wenn explos.

#             sign_total =sign_colour* sign_trace

#             EWs *= (1+sign_total*2*iso_perc/100)

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

        #print 'system hns and trace :\n',self.MT.get_eigvals(),N.sum(self.MT.get_eigvals()),'\n\n'
        #print 'system hns_devi and trace :\n',[EWh_devi,EWn_devi,EWs_devi],N.sum([EWh_devi,EWn_devi,EWs_devi]),'\n\n'
#         eigs,dummy = N.linalg.eigh(N.matrix(self.MT._M))
#         eigs_d, dum=N.linalg.eigh(N.matrix(self.MT._deviatoric))
#         print '\n e1   e3   E1   E2   E3 \n', N.round(eigs_d[0] ), N.round(eigs_d[2]), N.round(eigs[0]), N.round(eigs[1]), N.round(eigs[2])
#         print
        #exit()
        
        if abs(EWn) < epsilon:
            EWn = 0
        norm_factor =  max(N.abs([EWh, EWn, EWs]))   

        #print 'effective system hns and trace :\n',[EWh,EWn,EWs],EWh+EWn+EWs,'\n\n'
        [EWh, EWn, EWs] = [xx /norm_factor  for xx in [EWh, EWn, EWs] ]
        #print 'normiertes system hns and trace :\n',[EWh,EWn,EWs],EWh+EWn+EWs,'\n\n'
        

        RHS   = -EWs /(EWn * N.cos(phi)**2 + EWh * N.sin(phi)**2 )
       #  for i,grad in enumerate(phi):
#             if abs(360-grad) < 1 or abs(0-grad) < 1 or abs(90-grad) < 1 :
#                 RHS[i] -= 0.5
                
        if N.all([N.sign(xx)>=0 for xx in RHS ]):
            alpha = N.arctan(N.sqrt(RHS) ) *rad2deg 
        else:
            alpha = phi.copy()
            alpha[:] = 90
            self._pure_isotropic = 1

        #print alpha[:10]
        #print self._plot_isotropic_part,self._pure_isotropic
        #print '\n\n'
        #exit()

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
        
        line_tuple_pos = N.zeros((3, n_curve_points ))
        line_tuple_neg = N.zeros((3, n_curve_points ))
        

        for ii in N.arange(n_curve_points):
            pos_vec_in_EV_basis  = N.array([H_values[ii], N_values[ii], S_values_positive[ii] ]).transpose()
            neg_vec_in_EV_basis  = N.array([H_values[ii], N_values[ii], S_values_negative[ii] ]).transpose()
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
        projection = self.validate_projection(self._plot_projection)
        projections[projection](self)
        
    @staticmethod
    def _check_basis(basis, available):
        if basis not in available:
            raise MTError('basis %s not vailable for given plotting projection - '
                    + 'choose from: %s' % ', '.join(available))

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

        self._check_basis(self._plot_basis, ['NED'])

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

        self._check_basis(self._plot_basis, ['NED'])
 
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

        self._check_basis(self._plot_basis, ['NED'])

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

        self._check_basis(self._plot_basis, ['NED'])

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
    def _sort_curve_points(self, curve):
        """
        Checks, if curve points are in right order for line plotting.

        If not, a re-arranging is carried out.

        """

        
        sorted_curve = N.zeros((2,len(curve[0,:])))

        #in polar coordinates
        #
        r_phi_curve = N.zeros((len(curve[0,:]), 2))
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
            #logger.debug( 'corrected position of first point in curve to (%.2f,%.2f)\n'%(sorted_curve[0,0],sorted_curve[1,0]) ) 


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
        #mask_neg_in_pos = points_inside_poly(lo_points_in_neg_curve,lo_points_in_pos_curve)
        mask_neg_in_pos = 0
        for neg_point in lo_points_in_neg_curve:
            #mask_neg_in_pos *=  self._point_inside_polygon(neg_point[0], neg_point[1],lo_points_in_pos_curve )   
            mask_neg_in_pos +=  self._pnpoly(N.array(lo_points_in_pos_curve),N.array([neg_point[0], neg_point[1]]) )   
            #if self._pnpoly(N.array(lo_points_in_pos_curve),N.array([neg_point[0], neg_point[1]]) )  == 0:
            #    print neg_point
        if mask_neg_in_pos > len(lo_points_in_neg_curve)-3:
            #logger.debug( 'negative curve completely within positive curve')
            self._plot_curve_in_curve  =   1
        
        # check, if positive curve completely within negative curve
        #mask_pos_in_neg = points_inside_poly(lo_points_in_pos_curve,lo_points_in_neg_curve)
        mask_pos_in_neg = 0
        for pos_point in lo_points_in_pos_curve:
            #mask_pos_in_neg *=  self._point_inside_polygon(pos_point[0], pos_point[1],lo_points_in_neg_curve )
            mask_pos_in_neg +=  self._pnpoly(N.array(lo_points_in_neg_curve),N.array([pos_point[0], pos_point[1]]) )   
        #exit()
        if mask_pos_in_neg > len(lo_points_in_pos_curve)-3:
            #logger.debug('positive curve completely within negative curve')
            self._plot_curve_in_curve  =   -1

        #print 'curve in curve', self._plot_curve_in_curve
        #self._plot_curve_in_curve  =   1
        #exit()
        #print 'rotation matrix: ',self.MT._rotation_matrix
        
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
        Generates the final plot of the beachball projection on the unit sphere.


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
                raise MTError('saving of plot failed')

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
        #ax.plot( neg_nodalline[0,:] ,neg_nodalline[1,:],'go')

        ax.plot( pos_nodalline[0,:] ,pos_nodalline[1,:],c=self._plot_nodalline_colour,ls='-',lw=self._plot_nodalline_width, alpha=self._plot_nodalline_alpha*self._plot_total_alpha)

        
 
        if self._plot_show_faultplanes:

            ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha) 

            ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha) 

        elif self._plot_show_1faultplane:
            
            if self._plot_show_FP_index == 1:
                ax.plot( FP1_2_plot[0,:], FP1_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha) 

            elif self._plot_show_FP_index == 2:
                ax.plot( FP2_2_plot[0,:], FP2_2_plot[1,:],c=self._plot_faultplane_colour,ls='-',lw=self._plot_faultplane_width, alpha=self._plot_faultplane_alpha*self._plot_total_alpha) 
            else:
                raise MTError('unavailable fault plane index: %i' % _plot_show_FP_index)

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



        #plot 4 fake points, guaranteeing full visibilty of the sphere
        ax.plot([0,1.05,0,-1.05],[1.05,0,-1.05,0],',',alpha=0.)

        #scaling behaviour
        ax.autoscale_view(tight=True, scalex=True, scaley=True)


        return plotfig 

