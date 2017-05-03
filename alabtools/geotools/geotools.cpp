#include <CGAL/Simple_cartesian.h> 
#include <CGAL/algorithm.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <vector>

typedef CGAL::Simple_cartesian<CGAL::Gmpq>                   Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel>                       Nef_polyhedron;
typedef Kernel::Point_3                                      Point_3;
typedef Polyhedron::Vertex_iterator                          Vertex_iterator;

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
