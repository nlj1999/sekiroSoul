
#include "sekiroSoul.h"

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );


rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtBuffer<uchar4, 2>              output_buffer;

RT_PROGRAM void pinhole_camera()
{
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) / make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	optix::Ray ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(top_object, ray, prd);

	output_buffer[launch_index] = make_color(prd.result);
}


rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	prd_radiance.result = make_float3(tex2D(envmap, u, v));
}


RT_PROGRAM void any_hit_shadow()
{
	prd_shadow.attenuation = make_float3(0);

	rtTerminateRay();
}



rtDeclareVariable(float, metalKa, , ) = 1;
rtDeclareVariable(float, metalKs, , ) = 1;
rtDeclareVariable(float, metalroughness, , ) = .1;
rtDeclareVariable(float, rustKa, , ) = 1;
rtDeclareVariable(float, rustKd, , ) = 1;
rtDeclareVariable(float3, metalcolor, , ) = { .7, .7, .7 };
rtDeclareVariable(float3, ambient_light_color, , );
rtBuffer<BasicLight>       lights;
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );
rtDeclareVariable(float3, reflectivity_n, , );
#define MAXOCTAVES 6


RT_PROGRAM void metal_closest_hit_radiance()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 hit_point = ray.origin + t_hit * ray.direction;


	float3 color = metalcolor * metalKa * ambient_light_color;
	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nmDl = dot(ffnormal, L);

		if (nmDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += rustKd * nmDl * Lc;

				float r = nmDl;
				if (nmDl > 0.0f) {
					float3 H = normalize(L - ray.direction);
					float nmDh = dot(ffnormal, H);
					if (nmDh > 0)
						color += r * metalKs * Lc * pow(nmDh, 1.f / metalroughness);
				}
			}
		}
	}

	float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n);
	float importance = prd_radiance.importance * optix::luminance(r);

	// reflection ray
	if (importance > importance_cutoff && prd_radiance.depth < max_depth) {
		PerRayData_radiance refl_prd;
		refl_prd.importance = importance;
		refl_prd.depth = prd_radiance.depth + 1;
		float3 R = reflect(ray.direction, ffnormal);
		optix::Ray refl_ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refl_ray, refl_prd);
		color += r * refl_prd.result;
	}

	prd_radiance.result = color;
}


rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(float3, reflectivity, , );
rtDeclareVariable(float3, tile_v0, , );
rtDeclareVariable(float3, tile_v1, , );
rtDeclareVariable(float3, crack_color, , );
rtDeclareVariable(float, crack_width, , );

RT_PROGRAM void floor_closest_hit_radiance()
{
	float3 world_geo_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_geo_normal);
	float3 color = Ka * ambient_light_color;

	float3 hit_point = ray.origin + t_hit * ray.direction;

	float v0 = dot(tile_v0, hit_point);
	float v1 = dot(tile_v1, hit_point);
	v0 = v0 - floor(v0);
	v1 = v1 - floor(v1);

	float3 local_Kd;
	if (v0 > crack_width && v1 > crack_width) {
		local_Kd = Kd;
	}
	else {
		local_Kd = crack_color;
	}

	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);

		if (nDl > 0.0f) {
			// cast shadow ray
			PerRayData_shadow shadow_prd;
			shadow_prd.attenuation = make_float3(1.0f);
			float Ldist = length(light.pos - hit_point);
			optix::Ray shadow_ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist);
			rtTrace(top_shadower, shadow_ray, shadow_prd);
			float3 light_attenuation = shadow_prd.attenuation;

			if (fmaxf(light_attenuation) > 0.0f) {
				float3 Lc = light.color * light_attenuation;
				color += local_Kd * nDl * Lc;

				float3 H = normalize(L - ray.direction);
				float nDh = dot(ffnormal, H);
				if (nDh > 0)
					color += Ks * Lc * pow(nDh, phong_exp);
			}

		}
	}

	float3 r = schlick(-dot(ffnormal, ray.direction), reflectivity_n);
	float importance = prd_radiance.importance * optix::luminance(r);

	// reflection ray
	if (importance > importance_cutoff && prd_radiance.depth < max_depth) {
		PerRayData_radiance refl_prd;
		refl_prd.importance = importance;
		refl_prd.depth = prd_radiance.depth + 1;
		float3 R = reflect(ray.direction, ffnormal);
		optix::Ray refl_ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon);
		rtTrace(top_object, refl_ray, refl_prd);
		color += r * refl_prd.result;
	}

	prd_radiance.result = color;
}

rtDeclareVariable(float3, particle_color, attribute particle_color, );
rtDeclareVariable(float, particle_opacity, attribute particle_opacity, );
rtDeclareVariable(float3, shadow_attenuation, , );


//
// Dielectric surface shader
//
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(float, fresnel_exponent, , );
rtDeclareVariable(float, fresnel_minimum, , );
rtDeclareVariable(float, fresnel_maximum, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(int, refraction_maxdepth, , );
rtDeclareVariable(int, reflection_maxdepth, , );
rtDeclareVariable(float3, refraction_color, , );
rtDeclareVariable(float3, reflection_color, , );
rtDeclareVariable(float3, extinction_constant, , );
RT_PROGRAM void fire_closest_hit_radiance()
{
	/*
	// intersection vectors
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
	const float3 i = ray.direction;                                            // incident direction

	float reflection = 1.0f;
	float3 result = make_float3(0.0);

	float3 beer_attenuation;
	if (dot(n, ray.direction) > 0) {
		// Beer's law attenuation
		beer_attenuation = exp(extinction_constant * t_hit);
	}
	else {
		beer_attenuation = make_float3(1);
	}

	// refraction
	if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
	{
		float3 t;                                                            // transmission direction
		if (refract(t, i, n, refraction_index))
		{

			// check for external or internal reflection
			float cos_theta = dot(i, n);
			if (cos_theta < 0.0f)
				cos_theta = -cos_theta;
			else
				cos_theta = dot(t, n);

			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

			float importance = prd_radiance.importance * (1.0f - reflection) * optix::luminance(refraction_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, t, RADIANCE_RAY_TYPE, scene_epsilon);
				PerRayData_radiance refr_prd;
				refr_prd.depth = prd_radiance.depth + 1;
				refr_prd.importance = importance;

				rtTrace(top_object, ray, refr_prd);
				result += (1.0f - reflection) * refraction_color * refr_prd.result;
			}
			else {
				result += (1.0f - reflection) * refraction_color * cutoff_color;
			}
		}
		// else TIR
	}
	// reflection
	if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
	{
		float3 r = reflect(i, n);

		float importance = prd_radiance.importance * reflection * optix::luminance(reflection_color * beer_attenuation);
		if (importance > importance_cutoff) {
			optix::Ray ray(h, r, RADIANCE_RAY_TYPE, scene_epsilon);
			PerRayData_radiance refl_prd;
			refl_prd.depth = prd_radiance.depth + 10;
			refl_prd.importance = importance;

			rtTrace(top_object, ray, refl_prd);
			result += reflection * reflection_color * refl_prd.result;
		}
		else {
			result += reflection * reflection_color * cutoff_color;
		}
	}

	result = result * beer_attenuation;
	prd_radiance.result = result;*/
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	float opa = max(0.0f, particle_opacity);
	if (prd_radiance.depth < max_depth) {
		optix::Ray ray(h, ray.direction, RADIANCE_RAY_TYPE, scene_epsilon);
		PerRayData_radiance through;
		through.depth = prd_radiance.depth + 10;
		rtTrace(top_object, ray, through);
		prd_radiance.result = through.result * (1 - opa) + particle_color * opa;
	}
}

RT_PROGRAM void exception()
{
	output_buffer[launch_index] = make_color(bad_color);
}
