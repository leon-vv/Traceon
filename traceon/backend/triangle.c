#include <assert.h>
#include <time.h>
#include <math.h>


EXPORT int triangle_orientation_is_equal(int triangle_index1, int triangle_index2, uint64_t triangles[][3], double points[][3]) {
    // Imagine standing at a common vertex, and looking along the common edge
    // One way to define a normal, would be to draw a line from your position
    // to the midpoint, and taking a cross product with the edge.
    // In this way, we can define normals that point in the same direction.
    // To check if the normals that follow from the ordering of the triangles
    // agree, we can compare the normals as computed with the procedure explained.

    uint64_t *t1 = triangles[triangle_index1];
    uint64_t *t2 = triangles[triangle_index2];

    double triangle1[3][3] = { {points[t1[0]][0], points[t1[0]][1], points[t1[0]][2]},
                               {points[t1[1]][0], points[t1[1]][1], points[t1[1]][2]},
                               {points[t1[2]][0], points[t1[2]][1], points[t1[2]][2]} };

    double triangle2[3][3] = { {points[t2[0]][0], points[t2[0]][1], points[t2[0]][2]},
                               {points[t2[1]][0], points[t2[1]][1], points[t2[1]][2]},
                               {points[t2[2]][0], points[t2[2]][1], points[t2[2]][2]} };

    double mid1[3] = {(triangle1[0][0] + triangle1[1][0] + triangle1[2][0]) / 3.0,
                      (triangle1[0][1] + triangle1[1][1] + triangle1[2][1]) / 3.0,
                      (triangle1[0][2] + triangle1[1][2] + triangle1[2][2]) / 3.0};

    double mid2[3] = {(triangle2[0][0] + triangle2[1][0] + triangle2[2][0]) / 3.0,
                      (triangle2[0][1] + triangle2[1][1] + triangle2[2][1]) / 3.0,
                      (triangle2[0][2] + triangle2[1][2] + triangle2[2][2]) / 3.0};

    double normal1[3], normal2[3];
    normal_3d(triangle1, normal1);
    normal_3d(triangle2, normal2);

    // Find the common vertices between the two triangles
	int common_index1, common_index2;
	bool common_index_found = false;
	
    for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++) {
		if (t1[i] == t2[j]) {
			common_index1 = i;
			common_index2 = j;
			common_index_found = true;
		}
	}
	
	if(!common_index_found) return -1;
	
    double *common_vertex = points[t1[common_index1]];
	
	// Find the common edge
	double edges[4][3];
	int edge_count = 0;
	
	for(int i = 0; i < 3; i++) {
		if(i == common_index1) continue;
		edges[edge_count][0] = points[t1[i]][0] - common_vertex[0];
		edges[edge_count][1] = points[t1[i]][1] - common_vertex[1];
		edges[edge_count][2] = points[t1[i]][2] - common_vertex[2];
		edge_count += 1;
	}
	for(int j = 0; j < 3; j++) {
		if(j == common_index2) continue;
		edges[edge_count][0] = points[t2[j]][0] - common_vertex[0];
		edges[edge_count][1] = points[t2[j]][1] - common_vertex[1];
		edges[edge_count][2] = points[t2[j]][2] - common_vertex[2];
		edge_count += 1;
	}

	assert(edge_count == 4);

	for(int e = 0; e < 4; e++) normalize_3d(&edges[e][0]);
	
	// Two edges are equal if their dot product is one
	double *common_edge = NULL;

	for(int i = 0; i < 2; i++)
	for(int j = 0; j < 2; j++) {
		double d = dot_3d(edges[i], edges[2 + j]);
		if(fabs(d - 1.0) < 1e-14) common_edge = &edges[i][0];
	}

	if(common_edge == NULL) return -1;
		
    double to_mid1[3] = {mid1[0] - common_vertex[0], mid1[1] - common_vertex[1], mid1[2] - common_vertex[2]};
    double to_mid2[3] = {mid2[0] - common_vertex[0], mid2[1] - common_vertex[1], mid2[2] - common_vertex[2]};

    double upward_normal1[3], upward_normal2[3];
    cross_product_3d(to_mid1, common_edge, upward_normal1);
    cross_product_3d(common_edge, to_mid2, upward_normal2);

    double dot1 = dot_3d(upward_normal1, normal1);
    double dot2 = dot_3d(upward_normal2, normal2);

    return (dot1 > 0.0) == (dot2 > 0.0);
}












