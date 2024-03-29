/*
  Copyright (C) 2005 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

/*
 *  AUTHOR
 *    Catherine Loader, catherine@research.bell-labs.com.
 *    October 23, 2000.
 *
 *  Merge in to R:
 *	Copyright (C) 2000, The R Core Development Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
 *
 *
 *  DESCRIPTION
 *
 *    The density function of the F distribution.
 *    To evaluate it, write it as a Binomial probability with p = x*m/(n+x*m).
 *    For m >= 2, we use the simplest conversion.
 *    For m < 2, (m-2)/2 < 0 so the conversion will not work, and we must use
 *               a second conversion.
 *    Note the division by p; this seems unavoidable
 *    for m < 2, since the F density has a singularity as x (or p) -> 0.
 */

#include "nmath.hpp"
#include "dpq.hpp"
namespace Rmath{

double df(double x, double m, double n, int give_log)
{
    double p, q, f, dens;

#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(m) || ISNAN(n))
	return x + m + n;
#endif
    if (m <= 0 || n <= 0) ML_ERR_return_NAN;
    if (x <= 0.) return(R_D__0);

    f = 1./(n+x*m);
    q = n*f;
    p = x*m*f;

    if (m >= 2) {
	f = m*q/2;
	dens = dbinom_raw((m-2)/2, (m+n-2)/2, p, q, give_log);
    }
    else {
	f = m*m*q / (2*p*(m+n));
	dens = dbinom_raw(m/2, (m+n)/2, p, q, give_log);
    }
    return(give_log ? log(f)+dens : f*dens);
}
}

