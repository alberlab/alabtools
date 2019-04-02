#include <CGAL/Cartesian_d.h>
#include <CGAL/Random.h>
#include <CGAL/Exact_rational.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <vector>

typedef float                             FT;
typedef CGAL::Cartesian_d<FT>             K;
typedef CGAL::Min_sphere_of_spheres_d_traits_d<K,FT,3> Traits;
typedef CGAL::Min_sphere_of_spheres_d<Traits> Min_sphere;
typedef K::Point_d                        Point;
typedef Traits::Sphere                    Sphere;

using namespace std;

void bounding_spheres(float* crd, float* radii, int n_bead, int n_struct, float* results) {
  int offs;
  for (int i=0; i<n_struct; ++i) {
    std::vector<Sphere> S;
    for (int j=0; j<n_bead; ++j) {
      offs = (n_struct * j + i) * 3;
      Point p(3, &crd[offs], &crd[offs + 3]);
      S.push_back(Sphere(p, radii[j]));
    }
    Min_sphere ms(S.begin(),S.end());
    int k = 0;
    for (auto it = ms.center_cartesian_begin(); it != ms.center_cartesian_end(); ++it){
      results[i*4 + k] = *it;
      ++k;
    }
    results[i*4 + 3] = ms.radius();
  }
}

