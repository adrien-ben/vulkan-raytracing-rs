#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT bool isShadowed;
hitAttributeEXT vec2 attribs;

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec3 color;
};

struct Material {
    vec4 baseColor;
};

struct GeometryInfo {
    mat4 transform;
    Material material;
    uint vertexOffset;
    uint indexOffset;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3, set = 0) readonly buffer Vertices { Vertex v[]; } vertices;
layout(binding = 4, set = 0) readonly buffer Indices { uint i[]; } indices;
layout(binding = 5, set = 0) readonly buffer GeometryInfos { GeometryInfo g[]; } geometryInfos;

void main() {
    GeometryInfo geometryInfo = geometryInfos.g[gl_GeometryIndexEXT];

    // Fetch vertices
    uint vertexOffset = geometryInfo.vertexOffset;
    uint indexOffset = geometryInfo.indexOffset + (3 * gl_PrimitiveID);

    uint i0 = vertexOffset + indices.i[indexOffset];
    uint i1 = vertexOffset + indices.i[indexOffset + 1];
    uint i2 = vertexOffset + indices.i[indexOffset + 2];

    Vertex v0 = vertices.v[i0];
	Vertex v1 = vertices.v[i1];
	Vertex v2 = vertices.v[i2];

    // Interpolate and transform normal
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
    normal = normalize(geometryInfo.transform * vec4(normal, 0.0)).xyz;

    // Interpolate Color
    vec3 vertexColor = normalize(v0.color * barycentricCoords.x + v1.color * barycentricCoords.y + v2.color * barycentricCoords.z);
    vec3 baseColor = geometryInfo.material.baseColor.xyz;
    vec3 color = vertexColor * baseColor;


    // Lighting
    const vec3 lightColor = vec3(1.0);
    const vec3 lightDir = normalize(vec3(-2.0, -3.0, -2.0));
    float dot_prod = dot(-lightDir, normal);
    float factor = max(0.2, dot_prod);
    vec3 finalColor = factor * color * lightColor;

    hitValue = finalColor;

    if (dot_prod > 0) {
        // Shadow casting
        float tmin = 0.001;
        float tmax = 10.0;
        vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        isShadowed = true;

        // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
        const uint missIndex = 1;

        traceRayEXT(
            topLevelAS, 
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 
            0xFF, 
            0, 0, 
            missIndex, 
            origin, 
            tmin, 
            -lightDir, 
            tmax, 
            1
        );

        if (isShadowed) {
            hitValue *= 0.3;
        }
    }
}
