""" extrapolate y,m,d data with scipy UnivariateSpline """
import numpy as np
from scipy.interpolate import UnivariateSpline
    # pydoc scipy.interpolate.UnivariateSpline -- fitpack, unclear
from datetime import date
from pylab import *  # ipython -pylab

__version__ = "denis 23oct"


def daynumber( y,m,d ):
    """ 2005,1,1 -> 0  2006,1,1 -> 365 ... """
    return date( y,m,d ).toordinal() - date( 2005,1,1 ).toordinal()

days, values = np.array([
    (daynumber(2005,1,1), 1.2 ),
    (daynumber(2005,4,1), 1.8 ),
    (daynumber(2005,9,1), 5.3 ),
    (daynumber(2005,10,1), 5.3 )
    ]).T
dayswanted = np.array([ daynumber( year, month, 1 )
        for year in range( 2005, 2006+1 )
        for month in range( 1, 12+1 )])

np.set_printoptions( 1 )  # .1f
print("days:", days)
print("values:", values)
print("dayswanted:", dayswanted)

title( "extrapolation with scipy.interpolate.UnivariateSpline" )
plot( days, values, "o" )
for k in (1,2,3):  # line parabola cubicspline
    extrapolator = UnivariateSpline( days, values, k=k )
    y = extrapolator( dayswanted )
    label = "k=%d" % k
    print(label, y)
    plot( dayswanted, y, label=label  )  # pylab

legend( loc="lower left" )
grid(True)
savefig( "extrapolate-UnivariateSpline.png", dpi=50 )
show()