#include <CGAL/Simple_cartesian.h> 
#include <CGAL/algorithm.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

#include <CGAL/Cartesian_d.h>
#include <CGAL/Random.h>
#include <CGAL/Exact_rational.h>
#include <CGAL/Min_sphere_of_spheres_d.h>

#include <vector>
#include <omp.h>
#define THREADS 4

//For Convex Hull
typedef CGAL::Simple_cartesian<CGAL::Gmpq>                   Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel>                       Nef_polyhedron;
typedef Kernel::Point_3                                      Point_3;
typedef Polyhedron::Vertex_iterator                          Vertex_iterator;

//For Bounding Sphere
typedef float                             FT;
typedef CGAL::Cartesian_d<FT>             K;
typedef CGAL::Min_sphere_of_spheres_d_traits_d<K,FT,3> Traits;
typedef CGAL::Min_sphere_of_spheres_d<Traits> Min_sphere;
typedef K::Point_d                        Point;
typedef Traits::Sphere                    Sphere;



/*
 * A, B : 1D array of points unraveled from 2D array
 * 
 * Asize, Bsize : length of A,B
 * 
 * output : 1D vector of points of convex hull intersection unraveled from 2D array.
 * 
 */

std::vector<double> ConvexHullIntersection(double * A, int ASize,
                                           double * B, int BSize)
{
    std::vector<Point_3> pointsA, pointsB;
    
    for (int i = 0; i < int(ASize/3); ++i){
        pointsA.push_back(Point_3(A[i*3], A[i*3+1], A[i*3+2]));
    }
    for (int i = 0; i < int(BSize/3); ++i){
        pointsB.push_back(Point_3(B[i*3], B[i*3+1], B[i*3+2]));
    }
    
    Polyhedron PA,PB,PI;
    
    CGAL::convex_hull_3(pointsA.begin(),pointsA.end(),PA);
    CGAL::convex_hull_3(pointsB.begin(),pointsB.end(),PB);
    
    Nef_polyhedron NA(PA);
    Nef_polyhedron NB(PB);
    
    Nef_polyhedron NI = NA * NB;
    
    if(NI.is_simple()){
        NI.convert_to_polyhedron(PI);
    }
    
    std::vector<double> result;
    for (Vertex_iterator v = PI.vertices_begin(); v != PI.vertices_end(); ++v){
        //std::cout << v->point().x() << ' ' << v->point().y() << ' ' << v->point().z() << std::endl;
        result.push_back(v->point().x().to_double());
        result.push_back(v->point().y().to_double());
        result.push_back(v->point().z().to_double());
    }
    
    return result;
}

/*
 * Compute minimum bounding spheres for a group of beads.
 *
 * Parameters
 * ----------
 * crds : A nbeads x nstruct x 3 coordinates vector
 * radii : Radii of the beads[float]
 */

void bounding_spheres(float* crd, float* radii, int n_bead, int n_struct, float* results) {
    
#pragma omp parallel num_threads(THREADS)
{
    #pragma omp for schedule(dynamic, 5)
    for (int i=0; i<n_struct; ++i) {
        std::vector<Sphere> S;
        int offs;
        
        for (int j=0; j<n_bead; ++j) {
            offs = (n_struct * j + i) * 3;
            Point p(3, &crd[offs], &crd[offs + 3]);
            S.push_back(Sphere(p, radii[j]));
        }
        
        Min_sphere ms(S.begin(),S.end());
        
        int k = 0;
        for (Min_sphere::Cartesian_const_iterator it = ms.center_cartesian_begin(); it != ms.center_cartesian_end(); ++it){
            results[i*4 + k] = *it;
            ++k;
        }
        results[i*4 + 3] = ms.radius();
    }
}

}

