
#include <GL/glew.h>
#include <GL/glut.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "common.h"
#include "random.h"
#include <sutil.h>
#include <Arcball.h>
#include <OptiXMesh.h>
#include <tinyobjloader/tiny_obj_loader.h>

#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdint.h>
#include <math.h>

using namespace optix;


struct ParticleFrameData {
	std::vector<float3> positions;
	std::vector<float3> velocities;
	std::vector<float3> colors;
	std::vector<float> life;
	std::vector<float> opacity;
};

struct ParticlesBuffer
{
	Buffer      positions;
	Buffer      velocities;
	Buffer      colors;
	Buffer		life;
	Buffer		opacity;
};

const char* const SAMPLE_NAME = "sekiroSoul";
const char* sekiroPtx;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width = 1080;
uint32_t       height = 720;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;
Group topGroup;
float3 m_min, m_max, m_center;

int            frame_number = 1;
int            sqrt_num_samples = 2;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

bool            play = false;
unsigned int    accumulation_frame = 0;
unsigned int    iterations_per_animation_frame = 1;
ParticlesBuffer buffers;
Geometry particleGeom;
GeometryGroup particleGroup;
const int num_particles = 1000;
const int num_new_particles = 16;
int lastUsedParticle = 0;
ParticleFrameData cacheEntry;


bool attacking;
unsigned int attackMode;
float attackStartTime;
const float attackAnimationTime = 0.75f;
Transform baseTrans1;
Transform DynamicTrans1;
Transform particleTrans;


//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadMesh(const std::string& f1);
void updateGeometry();
void swordTrigger();
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void swap(float &x, float &y) {
	float t = x;
	x = y;
	y = t;
}
void minmaxFresh() {
	if (m_min.x > m_max.x)
		swap(m_min.x, m_max.x);
	if (m_min.y > m_max.y)
		swap(m_min.y, m_max.y);
	if (m_min.z > m_max.z)
		swap(m_min.z, m_max.z);
	m_center = 0.5 * (m_min + m_max);
}
static inline float3 get_min(
	const float3& v1,
	const float3& v2)
{
	float3 result;
	result.x = std::min<float>(v1.x, v2.x);
	result.y = std::min<float>(v1.y, v2.y);
	result.z = std::min<float>(v1.z, v2.z);
	return result;
}


static inline float3 get_max(
	const float3& v1,
	const float3& v2)
{
	float3 result;
	result.x = std::max<float>(v1.x, v2.x);
	result.y = std::max<float>(v1.y, v2.y);
	result.z = std::max<float>(v1.z, v2.z);
	return result;
}

static void fillBuffers(
	const std::vector<float3>& positions,
	const std::vector<float3>& velocities,
	const std::vector<float3>& colors,
	const std::vector<float>& life,
	const std::vector<float>& opacity)
{
	buffers.positions->setSize(positions.size());
	float* pos = reinterpret_cast<float*> (buffers.positions->map());
	for (int i = 0, index = 0; i < static_cast<int>(positions.size()); ++i) {
		float3 p = positions[i];

		pos[index++] = p.x;
		pos[index++] = p.y;
		pos[index++] = p.z;
	}
	buffers.positions->unmap();

	buffers.velocities->setSize(velocities.size());
	float* vel = reinterpret_cast<float*> (buffers.velocities->map());
	for (int i = 0, index = 0; i < static_cast<int>(velocities.size()); ++i) {
		float3 v = velocities[i];

		vel[index++] = v.x;
		vel[index++] = v.y;
		vel[index++] = v.z;
	}
	buffers.velocities->unmap();

	buffers.colors->setSize(colors.size());
	float* col = reinterpret_cast<float*> (buffers.colors->map());
	for (int i = 0, index = 0; i < static_cast<int>(colors.size()); ++i) {
		float3 c = colors[i];

		col[index++] = c.x;
		col[index++] = c.y;
		col[index++] = c.z;
	}
	buffers.colors->unmap();

	buffers.life->setSize(life.size());
	float* rad = reinterpret_cast<float*> (buffers.life->map());
	for (int i = 0; i < static_cast<int>(life.size()); ++i) {
		rad[i] = life[i];
	}
	buffers.life->unmap();

	buffers.opacity->setSize(life.size());
	float* opa = reinterpret_cast<float*> (buffers.opacity->map());
	for (int i = 0; i < static_cast<int>(opacity.size()); ++i) {
		opa[i] = opacity[i];
	}
	buffers.opacity->unmap();
}

static inline float parseFloat(const char*& token)
{
	token += strspn(token, " \t");
	float f = (float)atof(token);
	token += strcspn(token, " \t\r");
	return f;
}
static float rand_range(float min, float max)
{
	static unsigned int seed = 0u;
	return min + (max - min) * rnd(seed);
}

Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


void registerExitHandler()
{
	atexit(destroyContext);
}

GeometryInstance createGround(Material material,float scale){

	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	std::string ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	parallelogram->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "bounds"));
	parallelogram->setIntersectionProgram(context->createProgramFromPTXString(ptx, "intersect"));

	const float extent = scale * 30.0f; 
	float3 anchor = make_float3(-extent*0.5f, 0.02f, -extent*0.5f);
	float3 v1 = make_float3(0.0f, 0.0f, extent);
	float3 v2 = make_float3(extent, 0.0f, 0.0f);
	float3 normal = cross(v2, v1);
	normal = normalize(normal);
	float d = dot(normal, anchor);
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	float4 plane = make_float4(normal, d);
	parallelogram["plane"]->setFloat(plane);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);
	parallelogram["anchor"]->setFloat(anchor);


	GeometryInstance instance = context->createGeometryInstance(parallelogram, &material, &material + 1);
	return instance;
}
Matrix4x4 rotateMatrix(float3 alpha) {
	float floatMx[] = {
		1, 0, 0, 0,
		0, cosf(alpha.x), -sin(alpha.x), 0,
		0, sinf(alpha.x), cosf(alpha.x), 0,
		0, 0, 0, 1
	};
	float floatMy[] = {
		cosf(alpha.y), 0, sinf(alpha.y), 0,
		0, 1, 0, 0,
		-sinf(alpha.y), 0, cosf(alpha.y), 0,
		0, 0, 0, 1
	};
	float floatMz[] = {
		cosf(alpha.z), -sinf(alpha.z), 0, 0,
		sinf(alpha.z), cosf(alpha.z), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};
	Matrix4x4 Mx(floatMx), My(floatMy), Mz(floatMz);
	return Mz * My * Mx;
}
Matrix4x4 rotateVertexMatrix(float3 alpha, float3 offset) {
	float floatM[] = {
		1.0f, 0.0f, 0.0f, offset.x,
		0.0f, 1.0f, 0.0f, offset.y,
		0.0f, 0.0f, 1.0f, offset.z,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	float floatMM[] = {
		1.0f, 0.0f, 0.0f, -offset.x,
		0.0f, 1.0f, 0.0f, -offset.y,
		0.0f, 0.0f, 1.0f, -offset.z,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	Matrix4x4 mm(floatM), mM(floatMM);
	return mm * rotateMatrix(alpha) * mM;
}
Matrix4x4 moveMatrix(float3 offset) {
	float floatM[] = {
		1.0f, 0.0f, 0.0f, offset.x,
		0.0f, 1.0f, 0.0f, offset.y,
		0.0f, 0.0f, 1.0f, offset.z,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	return Matrix4x4(floatM);
}
Transform moveTrans(float3 offset) {
	float floatM[] = {
		1.0f, 0.0f, 0.0f, offset.x,
		0.0f, 1.0f, 0.0f, offset.y,
		0.0f, 0.0f, 1.0f, offset.z,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	Matrix4x4 mm(floatM);
	Transform trans = context->createTransform();
	trans->setMatrix(false, mm.getData(), mm.inverse().getData());
	return trans;
}
Transform rotateTrans(float3 alpha) {
	Matrix4x4 Mans = rotateMatrix(alpha);
	Transform trans = context->createTransform();
	trans->setMatrix(false, Mans.getData(), Mans.inverse().getData());
	return trans;
}
Transform rotateVertexTrans(float3 alpha, float3 offset) {

	Matrix4x4 Mtrans = rotateVertexMatrix(alpha, offset);
	Transform trans = context->createTransform();
	trans->setMatrix(false, Mtrans.getData(), Mtrans.inverse().getData());
	return trans;
}
float3 transPoint(float3 point, Matrix4x4 mat) {
	return make_float3(mat * make_float4(point, 1.0f));
}
float3 movePoint(float3 point, float3 offset) {
	return transPoint( point, moveMatrix(offset));
}
Matrix4x4 skill0(float t) {
	//plain attack
	float k = 1.0 - 2 * fabs(t - 0.5);
	return rotateVertexMatrix(make_float3(-1.57f * k, 0, 0), m_max);
}
Matrix4x4 skill1(float t) {
	//rotate attack
	return rotateVertexMatrix(make_float3(0, 6.28 * t, 0), m_max);
}
Matrix4x4 skill2(float t) {
	//down and rotate
	if (t < 0.1)
		return skill0(t * 5);
	else if (t < 0.9)
		return skill1(1.25 * t - 0.125) * skill0(0.5);
	else return skill0(5*t-4.0f);

}
Matrix4x4 skill3(float t) {
	float k = 1.0 - 2 * fabs(t - 0.5);
	return rotateVertexMatrix(make_float3(0, (0.5-t) * 3.14f, 0), m_max)*rotateVertexMatrix(make_float3(-1.57f * k, 0,0), m_max)* rotateVertexMatrix(make_float3(0, (t-0.5)*3.14f, 0), m_max);
}
Matrix4x4 skillStarter(float t) {
	switch (attackMode) {
	case 0: {
		return skill0(t);
	}
	case 1: {
		return skill2(t);
	}
	case 2: {
		return skill3(t);
	}
	}
}
float3 ranfloat3() {
	float rColor1 = ((rand() % 100) / 100.0f);
	float rColor2 = ((rand() % 100) / 100.0f);
	float rColor3 = ((rand() % 100) / 100.0f);
	return make_float3(rColor1, rColor2, rColor3);
}

void swordTrigger() {
	if (!attacking) {
		attacking = true;
		attackMode = (attackMode + 1) % 3;
		attackStartTime = sutil::currentTime();
	}
}

void createContext()
{
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(1);
	context->setStackSize(4640);
	context->setMaxTraceDepth(31);

	context["max_depth"]->setInt(100);
	context["scene_epsilon"]->setFloat(1.e-4f);
	context["importance_cutoff"]->setFloat(0.01f);
	context["ambient_light_color"]->setFloat(0.31f, 0.33f, 0.28f);

	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 4 * width * height, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
	context["output_buffer"]->set(buffer);

	// Setup programs
	Program ray_gen_program = context->createProgramFromPTXString(sekiroPtx, "pinhole_camera");
	context->setRayGenerationProgram(0, ray_gen_program);

	Program exception_program = context->createProgramFromPTXString(sekiroPtx, "exception");
	context->setExceptionProgram(0, exception_program);
	context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	// Miss program
	context->setMissProgram(0, context->createProgramFromPTXString(sekiroPtx, "envmap_miss"));
	const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
	const std::string texpath = std::string(sutil::samplesDir()) + "/data/dawn.hdr";
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texpath, default_color));
	context["bg_color"]->setFloat(make_float3(0.34f, 0.55f, 0.85f));
	
;
}
void createBase() {
	//group
	topGroup = context->createGroup();
	topGroup->setAcceleration(context->createAcceleration("Trbvh"));

	GeometryGroup groundGroup = context->createGeometryGroup();
	groundGroup->setAcceleration(context->createAcceleration("NoAccel"));
	topGroup->addChild(groundGroup);


	//floor
	Material floor_matl = context->createMaterial();
	Program floor_ch = context->createProgramFromPTXString(sekiroPtx, "floor_closest_hit_radiance");
	floor_matl->setClosestHitProgram(0, floor_ch);
	Program floor_ah = context->createProgramFromPTXString(sekiroPtx, "any_hit_shadow");
	floor_matl->setAnyHitProgram(1, floor_ah);
	floor_matl["Ka"]->setFloat(0.3f, 0.3f, 0.1f);
	floor_matl["Kd"]->setFloat(194 / 255.f * .6f, 186 / 255.f * .6f, 151 / 255.f * .6f);
	floor_matl["Ks"]->setFloat(0.4f, 0.4f, 0.4f);
	floor_matl["reflectivity"]->setFloat(0.1f, 0.1f, 0.1f);
	floor_matl["reflectivity_n"]->setFloat(0.05f, 0.05f, 0.05f);
	floor_matl["phong_exp"]->setFloat(88);
	floor_matl["tile_v0"]->setFloat(0.25f, 0, .15f);
	floor_matl["tile_v1"]->setFloat(-.15f, 0, 0.25f);
	floor_matl["crack_color"]->setFloat(0.1f, 0.1f, 0.1f);
	floor_matl["crack_width"]->setFloat(0.02f);

	GeometryInstance floorIns = createGround(floor_matl, 50.0f);
	groundGroup->addChild(floorIns);

	context["top_object"]->set(topGroup);
	context["top_shadower"]->set(topGroup);
}
void loadMesh(const std::string& f1){

	GeometryGroup objGroup = context->createGeometryGroup();
	objGroup->setAcceleration(context->createAcceleration("Trbvh"));

	// Mesh
	Material box_matl = context->createMaterial();
	Program box_ch = context->createProgramFromPTXString(sekiroPtx, "metal_closest_hit_radiance");
	box_matl->setClosestHitProgram(0, box_ch);
	Program box_ah = context->createProgramFromPTXString(sekiroPtx, "any_hit_shadow");
	box_matl->setAnyHitProgram(1, box_ah);
	box_matl["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	box_matl["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	box_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	box_matl["phong_exp"]->setFloat(88);
	box_matl["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	OptiXMesh mesh;
	mesh.context = context;
	mesh.use_tri_api = use_tri_api;
	mesh.ignore_mats = ignore_mats;
	loadMesh(f1, mesh);

	GeometryInstance meshIns = mesh.geom_instance;
	meshIns->setMaterialCount(1);
	meshIns->setMaterial(0, box_matl);
	objGroup->addChild(meshIns);


	Matrix4x4 transMat = rotateVertexMatrix(make_float3(1.57f, 0, 0.0f), mesh.bbox_max);
	Transform rotateBase = context->createTransform();
	Transform particleBase = context->createTransform();
	rotateBase->setMatrix(false, transMat.getData(), transMat.inverse().getData());
	particleBase->setMatrix(false, transMat.getData(), transMat.inverse().getData());
	m_min = mesh.bbox_min;
	m_max = mesh.bbox_max;
	minmaxFresh();

	DynamicTrans1 = rotateTrans(make_float3(0, 0, 0));
	baseTrans1 = moveTrans(make_float3(0, 0, 0));

	rotateBase->setChild(objGroup);
	baseTrans1->setChild(rotateBase);
	DynamicTrans1->setChild(baseTrans1);

	particleBase = context->createTransform();
	particleBase->setMatrix(false, transMat.getData(), transMat.inverse().getData());
	particleBase->setChild(particleGroup);

	particleTrans = moveTrans(make_float3(0, 0, 0));
	particleTrans->setChild(particleBase);

	topGroup->addChild(particleTrans);
	topGroup->addChild(DynamicTrans1);

}

void updateGeometry() {
	attacking = false;
	float elapsedTime = sutil::currentTime() - attackStartTime;
	if (elapsedTime >= attackAnimationTime)
		return;
	float t = elapsedTime / attackAnimationTime;
	Matrix4x4 tRotate = skillStarter(t);
	DynamicTrans1->setMatrix(false, tRotate.getData(), tRotate.inverse().getData());
	particleTrans->setMatrix(false, tRotate.getData(), tRotate.inverse().getData());
	attacking = true;
}

void rotateGeometry(Transform trans, Matrix4x4 transMat) {
	float t[16];
	trans->getMatrix(false, t, 0);
	Matrix4x4 mat(t);
	mat = transMat * mat;
	trans->setMatrix(false, mat.getData(), mat.inverse().getData());
	topGroup->getAcceleration()->markDirty();
	topGroup->getContext()->launch(0, 0, 0);
}
void moveGeometry(Transform trans, float3 offset) {
	float t[16];
	trans->getMatrix(false, t, 0);
	Matrix4x4 mat(t);
	mat = moveMatrix(offset) * mat;
	trans->setMatrix(false, mat.getData(), mat.inverse().getData());
	topGroup->getAcceleration()->markDirty();
	topGroup->getContext()->launch(0, 0, 0);
}

void mainMove(float3 offset) {
	moveGeometry(baseTrans1, offset);
	camera_lookat = camera_lookat + offset;
	camera_eye = camera_eye + offset;
	camera_changed = true;
	m_min = movePoint(m_min, offset);
	m_max = movePoint(m_max, offset);
	minmaxFresh();
}
void mainRotate(float3 alpha) {
	Matrix4x4 transMat = rotateVertexMatrix(alpha, m_min);
	rotateGeometry(baseTrans1->getChild<Transform>(), transMat);
	m_min = transPoint(m_min, transMat);
	m_max = transPoint(m_max, transMat);
	minmaxFresh();
}

int FirstUnusedParticle() {
	for (int i = lastUsedParticle; i < num_particles; i++) {
		if (cacheEntry.life[i] <= 0.0f) {
			lastUsedParticle = i;
			return i;
		}
	}
	for (int i = 0; i < num_particles; i++) {
		if (cacheEntry.life[i] <= 0.0f) {
			lastUsedParticle = i;
			return i;
		}
	}
	lastUsedParticle = 0;
	return 0;
}
void respawnParticle(int i) {
	cacheEntry.colors[i] = make_float3(1.0, 0.0, 0.0);
	cacheEntry.opacity[i] = 1.0f;
	int n = 10;
	float Adj_value = 0.05f;
	float radius = 0.1f;
	float3 record = make_float3(0.0f);
	for (int y = 0; y < n; y++) {//生成高斯分布的粒子，中心多，外边少
		record.x += (2.0f * float(rand()) / float(RAND_MAX) - 1.0f);
		record.y += (2.0f * float(rand()) / float(RAND_MAX) - 1.0f);
	}
	record.x *= radius;
	record.y *= radius;
	record.z = 3.0f*(float(rand()) / float(RAND_MAX)) - 5.0f;
	cacheEntry.positions[i] = record + m_center;
	cacheEntry.velocities[i] = make_float3(0.5f * (float(rand()) / float(RAND_MAX)) + 0.5f);//在最大最小速度之间随机选择
	float dist = sqrt(record.x * record.x + record.z * record.z);
	cacheEntry.life[i] = (1.0f - 0.5f) * (float(rand()) / float(RAND_MAX)) + 0.5f;
}
void setupParticles() {
	// the buffers will be set to the right size at a later stage
	buffers.positions = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
	buffers.velocities = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
	buffers.colors = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
	buffers.life = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 0);
	buffers.opacity = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 0);

	particleGeom = context->createGeometry();
	particleGeom["positions_buffer"]->setBuffer(buffers.positions);
	particleGeom["velocities_buffer"]->setBuffer(buffers.velocities);
	particleGeom["colors_buffer"]->setBuffer(buffers.colors);
	particleGeom["opacity_buffer"]->setBuffer(buffers.opacity);
	//particleGeom["radii_buffer"]->setBuffer(buffers.life);
	const char* ptx = sutil::getPtxString(SAMPLE_NAME, "particles.cu");
	particleGeom->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "particle_bounds"));
	particleGeom->setIntersectionProgram(context->createProgramFromPTXString(ptx, "particle_intersect"));


	Program closest_hit;
	closest_hit = context->createProgramFromPTXString(sekiroPtx, "fire_closest_hit_radiance");

	std::vector<Material> optix_materials;
	Material mat = context->createMaterial();
	mat->setClosestHitProgram(0u, closest_hit);

	mat["importance_cutoff"]->setFloat(1e-2f);
	mat["cutoff_color"]->setFloat(0.34f, 0.55f, 0.85f);
	mat["fresnel_exponent"]->setFloat(3.0f);
	mat["fresnel_minimum"]->setFloat(0.1f);
	mat["fresnel_maximum"]->setFloat(1.0f);
	mat["refraction_index"]->setFloat(1.4f);
	mat["refraction_color"]->setFloat(1.0f, 1.0f, 1.0f);
	mat["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
	mat["refraction_maxdepth"]->setInt(100);
	mat["reflection_maxdepth"]->setInt(100);
	float3 extinction = make_float3(.80f, .89f, .75f);
	mat["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
	mat["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);

	optix_materials.push_back(mat);

	GeometryInstance geom_instance = context->createGeometryInstance(
		particleGeom,
		optix_materials.begin(),
		optix_materials.end());

	particleGroup = context->createGeometryGroup();
	particleGroup->addChild(geom_instance);
	particleGroup->setAcceleration(context->createAcceleration("Trbvh"));
	
	for (int i = 0; i < num_particles; i++) {
		float3 zero = make_float3(0, 0, 0);
		cacheEntry.colors.push_back(zero);
		cacheEntry.positions.push_back(zero);
		cacheEntry.velocities.push_back(zero);
		cacheEntry.life.push_back(0.0f);
		cacheEntry.opacity.push_back(0.0f);
	}

	particleGeom->setPrimitiveCount(num_particles);
	//topGroup->addChild(particleGroup);
}
void loadParticles() {

	for (int i = 0; i < num_new_particles; i++) {
		int t = FirstUnusedParticle();
		respawnParticle(t);
	}
	static float lastTime = sutil::currentTime();
	float dt = sutil::currentTime() - lastTime;
	lastTime += dt;
	int sum = 0;
	for (int i = 0; i < num_particles; i++) {
		cacheEntry.life[i] -= dt;
		if (cacheEntry.life[i] > 0.0f) {
			cacheEntry.positions[i] -= cacheEntry.velocities[i] * dt;
			cacheEntry.velocities[i] += make_float3(0, 1.0f, 1.0f) * dt;
			cacheEntry.opacity[i] -= 2.5 * dt;
			sum++;
		}
	}


	// fills up the buffers
	fillBuffers(cacheEntry.positions, cacheEntry.velocities, cacheEntry.colors, cacheEntry.life, cacheEntry.opacity);

	// builds the BVH (or re-builds it if already existing)
	Acceleration accel = particleGroup->getAcceleration();
	accel->markDirty();
}

void setupLights() {

	BasicLight lights[] = {
		{ make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.2f, 0.2f, 0.25f), 0, 0 },
		{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.1f, 0.1f, 0.10f), 0, 0 },
		{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.7f, 0.7f, 0.65f), 1, 0 }
	};
	lights[0].pos *= 5.0f;
	lights[1].pos *= 5.0f;
	lights[2].pos *= 5.0f;

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	context["lights"]->set(light_buffer);
}

void setupCamera()
{

	camera_eye = make_float3(10.f, 10.0f, 30.f);
	camera_lookat = m_center;
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}

void updateCamera()
{
	const float vfov = 60.0f;
	const float aspect_ratio = static_cast<float>(width) /
		static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	const Matrix4x4 frame = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans = frame * camera_rotate*camera_rotate*frame_inv;

	camera_eye = make_float3(trans*make_float4(camera_eye, 1.0f));
	camera_lookat = make_float3(trans*make_float4(camera_lookat, 1.0f));
	camera_up = make_float3(trans*make_float4(camera_up, 0.0f));

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = Matrix4x4::identity();

	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);

}

void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}
void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{

	if (play && accumulation_frame >= iterations_per_animation_frame)
		loadParticles();
	context["frame"]->setUint(accumulation_frame++);
	topGroup->getAcceleration()->markDirty();
	topGroup->getContext()->launch(0, 0, 0);
	updateCamera();
	updateGeometry();
	context->launch(0, width, height);

	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{

	switch (k)
	{
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('q'):
	{
		camera_eye = camera_eye + make_float3(0, 0, 1.0f);
		camera_changed = true;
		break;
	}
	case('e'):
	{
		camera_eye = camera_eye - make_float3(0, 0, 1.0f);
		camera_changed = true;
		break;
	}
	case('s'):
	{
		camera_eye = camera_eye - make_float3(0, 1.0f, 0);
		camera_changed = true;
		break;
	}
	case('w'):
	{
		float3 dir = normalize(camera_lookat - camera_eye); 
		//mainRotate(dir);
		mainMove(dir);
		break;
	}
	case('a'):
	{
		camera_eye = camera_eye - make_float3(1.0f, 0, 0);
		camera_changed = true;
		break;
	}
	case('d'):
	{
		camera_eye = camera_eye + make_float3(1.0f, 0, 0);
		camera_changed = true;
		break;
	}
	case(' '):
	{
		swordTrigger();
		break;
	}
	case ('p'):
	{
		play = !play;
		break;
	}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}


void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
		camera_changed = true;
	}
	else if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
							  static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
							  static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, (from.y + (to.y - from.y)* 0.3f) / height };

		camera_rotate = arcball.rotate(b, a);
		camera_changed = true;
	}

	mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	camera_changed = true;

	width = w;
	height = h;
	sutil::ensureMinimumSize(width, height);

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------



int main(int argc, char** argv)
{
	std::string mesh_file = std::string(sutil::samplesDir()) + "/data/s1.obj"; 
	std::string particles_file = std::string(sutil::samplesDir()) + "/data/particles/particles.0001.txt";
	try
	{
		glutInitialize(&argc, argv);

		glewInit();

		sekiroPtx = sutil::getPtxString(SAMPLE_NAME, "sekiroSoul.cu");
		createContext();
		setupParticles();
		//loadParticles();
		createBase();
		loadMesh(mesh_file);
		setupCamera();
		setupLights();

		context->validate();

		glutRun();

		return 0;
	}
	SUTIL_CATCH(context->get())
}

