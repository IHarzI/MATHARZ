#ifndef MATHARZ_HEADER
#define MATHARZ_HEADER

// @2023-2024 (IHarzI) Maslianka Zakhar
// Written mainly for own game engine project(Phosphor engine)
// Header only, and easy to use math lib(at least for me, i hope), mainly for game(engine) projects
// WIP

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <initializer_list>

#define NOMINMAX 

#define MATHARZ_TRUE 1

#define MATHARZ_FALSE 0

#ifndef MATHARZ_INLINE
#define MATHARZ_INLINE inline
#endif // !MATHARZ_INLINE

// for custom handling of static, define this macro before including
#ifndef MATHARZ_STATIC_GLOBAL
#define MATHARZ_STATIC_GLOBAL static
#endif // !MATHARZ_STATIC_GLOBAL

// Default float point type for typedefs and other default stuff
#define MATHARZ_DEFAULT_FP float
#ifdef MATHARZ_USE_DOUBLE
#define MATHARZ_DEFAULT_FP double
#endif // MATHARZ_USE_DOUBLE

#ifdef MATHARZ_ASSERIONS_ENABLED
#ifdef _MSC_VER
#include <intrin.h>
#define DEBUG_BREAK() __debugbreak()
#else
#define DEBUG_BREAK() __builtin_trap()
#endif // DEBUG BREAK
#ifdef MATHARZ_CUSTOM_ASSERION // assertion must accept condition as bool and x argument as stream, assert << x << ......
#define MATHARZ_ASSERT(condition,x) MATHARZ_CUSTOM_ASSERION(condition,"[MATHARZ]: ASSERT: " << x);
#else
#include <iostream>
#define MATHARZ_DEFAULT_ASSERION
#define MATHARZ_ASSERT(condition,x) \
	if(!(condition)) {std::cout << "\n[MATHARZ]: ASSERT: " << x << '\n'; DEBUG_BREAK();}
#endif // MATHARZ_CUSTOM_ASSERION
#else
#define MATHARZ_ASSERT(condition, x)
#endif // MATHARZ_ASSERIONS_ENABLED

#define MATHARZ_CHECK(condition) \
MATHARZ_ASSERT((condition), "CONDITION FAILED!");

// Small enough value to check float with guard
#define MTHRZ_FLOAT_EPSILON 0.00001f

#define matharz harz::math
namespace harz {
	namespace math {


		// Forward declaration of types
		template<typename number_t>
		struct template_vec2;
		template<typename number_t>
		struct template_vec3;
		template<typename number_t>
		struct template_vec4;
		template<typename number_t>
		struct template_quaternion;
		template<typename number_t>
		struct template_matrix2x2;
		template<typename number_t>
		struct template_matrix3x3;
		template<typename number_t>
		struct template_matrix4x4;

		template<typename number_t>
		using template_row2 = template_vec2<number_t>;
		template<typename number_t>
		using template_row3 = template_vec3<number_t>;
		template<typename number_t>
		using template_row4 = template_vec4<number_t>;

		// TODO: column vectors
		template<typename number_t>
		struct template_column2;
		template<typename number_t>
		struct template_column3;
		template<typename number_t>
		struct template_column4;

		template<typename number_t>
		using template_matrix2x1 = template_column2<number_t>;
		template<typename number_t>
		using template_matrix3x1 = template_column3<number_t>;
		template<typename number_t>
		using template_matrix4x1 = template_column4<number_t>;

		// float 32 quat, by default
		using quat = template_quaternion<MATHARZ_DEFAULT_FP>;

		// float 32 vec2, by default
		using vec2 = template_vec2<MATHARZ_DEFAULT_FP>;
		// float64 vec2
		using vec2d = template_vec2<double>;
		// float32 vec2
		using vec2f = template_vec2<float>;
		// int32 vec2
		using vec2i = template_vec2<int>;
		// int16 vec2
		using vec2s = template_vec2<short>;
		// int8 vec2
		using vec2c = template_vec2<char>;

		// float 32 vec3, by default
		using vec3 = template_vec3<MATHARZ_DEFAULT_FP>;
		// float64 vec3
		using vec3d = template_vec3<double>;
		// float32 vec3
		using vec3f = template_vec3<float>;
		// int32 vec3
		using vec3i = template_vec3<int>;
		// int16 vec3
		using vec3s = template_vec3<short>;
		// int8 vec3
		using vec3c = template_vec3<char>;

		// float 32 vec4, by default
		using vec4 = template_vec4<MATHARZ_DEFAULT_FP>;
		// float64 vec4
		using vec4d = template_vec4<double>;
		// float32 vec4
		using vec4f = template_vec4<float>;
		// int32 vec4
		using vec4i = template_vec4<int>;
		// int16 vec4
		using vec4s = template_vec4<short>;
		// int8 vec4
		using vec4c = template_vec4<char>;

		// float 32 mat2, by default
		using mat2x2 = template_matrix2x2<MATHARZ_DEFAULT_FP>;
		// float64 mat2
		using mat2x2d = template_matrix2x2<double>;
		// float32 mat2
		using mat2x2f = template_matrix2x2<float>;
		// int32 mat2
		using mat2x2i = template_matrix2x2<int>;
		// int16 mat2
		using mat2x2s = template_matrix2x2<short>;
		// int8 mat2
		using mat2x2c = template_matrix2x2<char>;

		// float 32 mat3, by default
		using mat3x3 = template_matrix3x3<MATHARZ_DEFAULT_FP>;
		// float64 mat3
		using mat3x3d = template_matrix3x3<double>;
		// float32 mat3
		using mat3x3f = template_matrix3x3<float>;
		// int32 mat3
		using mat3x3i = template_matrix3x3<int>;
		// int16 mat3
		using mat3x3s = template_matrix3x3<short>;
		// int8 mat3
		using mat3x3c = template_matrix3x3<char>;

		// float 32 mat4, by default
		using mat4x4 = template_matrix4x4<MATHARZ_DEFAULT_FP>;
		// float64 mat4
		using mat4x4d = template_matrix4x4<double>;
		// float32 mat4
		using mat4x4f = template_matrix4x4<float>;
		// int32 mat4
		using mat4x4i = template_matrix4x4<int>;
		// int16 mat4
		using mat4x4s = template_matrix4x4<short>;
		// int8 mat4
		using mat4x4c = template_matrix4x4<char>;

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE number_t pi()
		{
			return number_t(3.1415926535897932384626433);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t two_pi()
		{
			return number_t(6.28318530717958647692528676655900576);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_pi()
		{
			return number_t(1.772453850905516027);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t half_pi()
		{
			return number_t(1.57079632679489661923132169163975144);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t three_over_two_pi()
		{
			return number_t(4.71238898038468985769396507491925432);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t quarter_pi()
		{
			return number_t(0.785398163397448309615660845819875721);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t one_over_pi()
		{
			return number_t(0.318309886183790671537767526745028724);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t one_over_two_pi()
		{
			return number_t(0.159154943091895335768883763372514362);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t two_over_pi()
		{
			return number_t(0.636619772367581343075535053490057448);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t four_over_pi()
		{
			return number_t(1.273239544735162686151070106980114898);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t two_over_root_pi()
		{
			return number_t(1.12837916709551257389615890312154517);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t one_over_root_two()
		{
			return number_t(0.707106781186547524400844362104849039);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_half_pi()
		{
			return number_t(1.253314137315500251);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_two_pi()
		{
			return number_t(2.506628274631000502);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_ln_four()
		{
			return number_t(1.17741002251547469);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t e()
		{
			return number_t(2.71828182845904523536);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t euler()
		{
			return number_t(0.577215664901532860606);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_two()
		{
			return number_t(1.41421356237309504880168872420969808);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_three()
		{
			return number_t(1.73205080756887729352744634150587236);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t root_five()
		{
			return number_t(2.23606797749978969640917366873127623);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t ln_two()
		{
			return number_t(0.693147180559945309417232121458176568);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t ln_ten()
		{
			return number_t(2.30258509299404568401799145468436421);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t ln_ln_two()
		{
			return number_t(-0.3665129205816643);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t third()
		{
			return number_t(0.3333333333333333333333333333333333333333);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t two_thirds()
		{
			return number_t(0.666666666666666666666666666666666666667);
		}

		template<typename number_t = MATHARZ_DEFAULT_FP>
		MATHARZ_INLINE  number_t golden_ratio()
		{
			return number_t(1.61803398874989484820458683436563811);
		}

		MATHARZ_INLINE float radians(float degrees)
		{
			return degrees * pi<float>() / 180.0;
		}

		MATHARZ_INLINE double radians(double degrees)
		{
			return degrees * pi<double>() / 180.0;
		}

		MATHARZ_INLINE long double radians(long double degrees)
		{
			return degrees * pi<long double>() / 180.0;
		}

		MATHARZ_INLINE float degrees(float radians)
		{
			return radians * 180.0 / pi<float>();
		}

		MATHARZ_INLINE double degrees(double radians)
		{
			return radians * 180.0 / pi<double>();
		}

		MATHARZ_INLINE long double degrees(long double radians)
		{
			return radians * 180.0 / pi<long double>();
		}

		MATHARZ_STATIC_GLOBAL constexpr float MaxFloat32()
		{
			return std::numeric_limits<float>::max();
		}

		MATHARZ_STATIC_GLOBAL constexpr double MaxFloat64()
		{
			return std::numeric_limits<double>::max();
		}

		template<typename number_t>
		MATHARZ_INLINE number_t infinity()
		{
			return std::numeric_limits<number_t>::infinity();
		}

		template<typename number_t, typename other_number_t>
		MATHARZ_INLINE number_t pow(number_t value, other_number_t exp)
		{
			return std::pow(value, exp);
		}

		MATHARZ_INLINE float pow(float value, float exp)
		{
			return std::powf(value, exp);
		}

		MATHARZ_INLINE double pow(double value, double exp)
		{
			return std::pow(value, exp);
		}

		template<typename number_t = float>
		MATHARZ_INLINE number_t sqrt(number_t value)
		{
			return std::sqrt(value);
		}

		MATHARZ_INLINE float sqrt(float value)
		{
			return std::sqrtf(value);
		}

		MATHARZ_INLINE double sqrt(double value)
		{
			return std::sqrt(value);
		}

		// Min value from x, y
		template<typename comparableType, typename other_comparableType>
		MATHARZ_INLINE comparableType min(comparableType x, other_comparableType y)
		{
			return (x < y) ? x : y;
		}

		MATHARZ_INLINE float min(float x, float y)
		{
			return std::fminf(x, y);
		}

		MATHARZ_INLINE double min(double x, double y)
		{
			return std::fmin(x, y);
		}

		// Max value from x, y
		template<typename comparableType, typename other_comparableType>
		MATHARZ_INLINE comparableType max(comparableType x, other_comparableType y)
		{
			return (x > y) ? x : y;
		}

		MATHARZ_INLINE float max(float x, float y)
		{
			return std::fmaxf(x, y);
		}

		MATHARZ_INLINE double max(double x, double y)
		{
			return std::fmax(x, y);
		}

		// Abs value x
		template<typename comparableType>
		MATHARZ_INLINE comparableType abs(comparableType x)
		{
			return std::abs(x);
		}

		// Abs value x
		MATHARZ_INLINE float abs(float x)
		{
			return std::fabsf(x);
		}

		// Abs value x
		MATHARZ_INLINE double abs(double x)
		{
			return std::fabs(x);
		}

		template<typename number_t>
		MATHARZ_INLINE template_vec2<number_t> abs(template_vec2<number_t> vec2)
		{
			return { abs(vec2.x), abs(vec2.y) };
		}

		template<typename number_t>
		MATHARZ_INLINE template_vec3<number_t> abs(template_vec3<number_t> vec3)
		{
			return { abs(vec3.x), abs(vec3.y), abs(vec3.z) };
		}

		template<typename number_t>
		MATHARZ_INLINE template_vec4<number_t> abs(template_vec4<number_t> vec4)
		{
			return { abs(vec4.x) , abs(vec4.y), abs(vec4.z), abs(vec4.w) };
		}

		// Remainder of division from x / y
		template<typename number_t, typename other_number_t>
		MATHARZ_INLINE number_t mod(number_t x, other_number_t y)
		{
			return x % y;
		}

		MATHARZ_INLINE int mod(int x, int y)
		{
			return x % y;
		}

		MATHARZ_INLINE float mod(float x, float y)
		{
			return std::fmodf(x, y);
		}

		MATHARZ_INLINE double mod(double x, double y)
		{
			return std::fmod(x, y);
		}

		MATHARZ_INLINE long double mod(long double x, long double y)
		{
			return std::fmod(x, y);
		}

		// Cos 
		template<typename number_t = float>
		MATHARZ_INLINE number_t cos(number_t x)
		{
			return std::cos(x);
		}


		MATHARZ_INLINE float cos(float x)
		{
			return std::cosf(x);
		}


		MATHARZ_INLINE double cos(double x)
		{
			return std::cos(x);
		}

		// Acos 
		template<typename number_t = float>
		MATHARZ_INLINE number_t acos(number_t x)
		{
			return std::acos(x);
		}

		MATHARZ_INLINE float acos(float x)
		{
			return std::acosf(x);
		}


		MATHARZ_INLINE double acos(double x)
		{
			return std::acos(x);
		}

		// Sin
		template<typename number_t = float>
		MATHARZ_INLINE number_t sin(number_t x)
		{
			return std::sin(x);
		}

		MATHARZ_INLINE float sin(float x)
		{
			return std::sinf(x);
		}


		MATHARZ_INLINE double sin(double x)
		{
			return std::sin(x);
		}

		// Asin
		template<typename number_t = float>
		MATHARZ_INLINE number_t asin(number_t x)
		{
			return std::asin(x);
		}

		MATHARZ_INLINE float asin(float x)
		{
			return std::asinf(x);
		}


		MATHARZ_INLINE double asin(double x)
		{
			return std::asin(x);
		}

		MATHARZ_INLINE float floor(float value)
		{
			return std::floor(value);
		}

		MATHARZ_INLINE double floor(double value)
		{
			return std::floor(value);
		}

		MATHARZ_INLINE long double floor(long double value)
		{
			return std::floor(value);
		}

		MATHARZ_INLINE float ceil(float value)
		{
			return std::ceil(value);
		}

		MATHARZ_INLINE double ceil(double value)
		{
			return std::ceil(value);
		}

		MATHARZ_INLINE long double ceil(long double value)
		{
			return std::ceil(value);
		}

		// Tan
		template<typename number_t = float>
		MATHARZ_INLINE number_t tan(number_t x)
		{
			return std::tan(x);
		}

		MATHARZ_INLINE float tan(float x)
		{
			return std::tanf(x);
		}


		MATHARZ_INLINE double tan(double x)
		{
			return std::tan(x);
		}

		// Atan2
		template<typename number_t = float>
		MATHARZ_INLINE number_t atan2(number_t y, number_t x)
		{
			return std::atan2(y, x);
		}

		MATHARZ_INLINE float atan2(float y, float x)
		{
			return std::atan2f(y, x);
		}


		MATHARZ_INLINE double atan2(double y, double x)
		{
			return std::atan2(y, x);
		}

		// Clamp value from x, y
		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE number_t clamp(number_t n, other_number_t lower, other_number_t upper)
		{
			// Simple and quick implementation, reference https://stackoverflow.com/a/9324086
			return max(lower, min(n, upper));
		}

		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE bool equal(number_t a, other_number_t b, float safeGuard = MTHRZ_FLOAT_EPSILON)
		{
			return abs(a - b) <= safeGuard;
		};

		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE bool more(number_t a, other_number_t b)
		{
			return a > b;
		};

		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE bool less(number_t a, other_number_t b)
		{
			return a < b;
		};

		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE bool more_or_equal(number_t a, other_number_t b, float safeGuard = MTHRZ_FLOAT_EPSILON)
		{
			return more(a, b) || equal(a, b);
		};

		template<typename number_t = float, typename other_number_t = float>
		MATHARZ_INLINE bool less_or_equal(number_t a, other_number_t b, float safeGuard = MTHRZ_FLOAT_EPSILON)
		{
			return less(a, b) || equal(a, b);
		};

		template<typename number_t = float, typename other_number_t1 = float, typename other_number_t2 = float>
		MATHARZ_INLINE bool less_or_more(number_t a, other_number_t1 lessThreshold, other_number_t1 moreThreshold)
		{
			return less(a, lessThreshold) || more(a, moreThreshold);
		};

		// Check if a is less/equal or more/equal then given less/more thresholds
		template<typename number_t = float, typename other_number_t1 = float, typename other_number_t2 = float>
		MATHARZ_INLINE bool less_or_more_equal(number_t a, other_number_t1 lessThreshold, other_number_t1 moreThreshold,
			float safe_guard = MTHRZ_FLOAT_EPSILON)
		{
			return less_or_equal(a, lessThreshold) || more_or_equal(a, moreThreshold);
		};

		// Ray Tracing MATH
		// -----------------

		// Schlick's approximation for reflections
		MATHARZ_INLINE double reflectance(double cosine, double refractionIndex) {
			auto r0 = (1 - refractionIndex) / (1 + refractionIndex);
			r0 = r0 * r0;
			return r0 + (1 - r0) * pow((1 - cosine), 5);
		}
		// -----------------

		// Vector with 2 components
		template<typename number_t>
		struct template_vec2 {
		public:
			union {
				union { number_t data[2]; };
				union { number_t rowData[1][2]; };
				union { number_t columnData[2][1]; };
				union { number_t lineArrayData[2]; };
				union { struct { number_t x, y; }; };
				union { struct { number_t r, g; }; };
				union { struct { number_t u, v; }; };
			};

			template_vec2<number_t>() { std::fill_n(&lineArrayData[0], 2, 0); };

			template_vec2<number_t>(number_t x, number_t y)
				: x(x), y(y) {};

			explicit template_vec2<number_t>(number_t data[2])
				: x(data[0]), y(data[1]) {};

			template_vec2<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 2, "Init list doesn't have a proper size(>2 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 2, *initList.begin());
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 2), data);
				};
			};

			// Cast to another vec_2 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_vec2< different_number_t>() const
			{
				return { static_cast<different_number_t>(this->x),static_cast<different_number_t>(this->y) };
			};

			MATHARZ_INLINE bool NearZero() const
			{
				return less(abs(*this), MTHRZ_FLOAT_EPSILON);
			}

			// Math functions
			MATHARZ_INLINE number_t SquareLength() const
			{
				return { x * x + y * y };
			}

			MATHARZ_INLINE number_t SquareMagnitude() const
			{
				return SquareLength();
			}

			MATHARZ_INLINE number_t SumOfElements() const
			{
				return x + y;
			}

			MATHARZ_INLINE number_t Lenght() const
			{
				return static_cast<number_t>(sqrt(SquareLength()));
			}

			MATHARZ_INLINE number_t Magnitude() const
			{
				return Lenght();
			}

			MATHARZ_INLINE number_t DotProductFromSelf() const
			{
				return SquareLength();
			}

			// Calculate dotProduct
			// @return dotProduct
			MATHARZ_INLINE number_t DotProduct(const template_vec2<number_t> b) const
			{
				return { x * b.x + y * b.y };
			}

			// Check if this vector is perpendicular to vector b
			MATHARZ_INLINE bool IsPerpendicularTo_INT(const template_vec2<number_t> b) const
			{
				return (DotProduct(b) == static_cast<number_t>(0));
			}

			// Check if this vector is parallel to vector b
			MATHARZ_INLINE bool IsParallelTo_INT(const template_vec2<number_t> b) const
			{
				return (CrossProduct(b) == static_cast<number_t>(0));
			}

			/*	Check if this vector is perpendicular to vector b with safe guard
				for avoiding bugs with floating point, (false)(0.0001 == 0) ; (true)(-safeGuard(.15) <0.0001 < safeGuard)
			*/
			MATHARZ_INLINE bool IsPerpendicularTo(const template_vec2<number_t> b, number_t safeGuard = 0.15) const
			{
				auto tempDP = DotProduct(b);
				return less_or_equal(abs(tempDP), safeGuard);
			}

			/*	Check if this vector is parallel to vector b with safe guard
				(for avoiding bugs with floating point, (false)(0.0001 == 0) ; (true)(-safeGuard(.15) <0.0001 < safeGuard))
			*/
			MATHARZ_INLINE bool IsParallelTo(const template_vec2<number_t> b, number_t safeGuard = 0.15) const
			{
				auto tempVector = CrossProduct(b);
				return less_or_equal(abs(tempVector), safeGuard);
			}

			// Calcualte distance to vector b
			MATHARZ_INLINE number_t DistanceTo(const template_vec2<number_t> b) const
			{
				template_vec2<number_t> DirectionVec = b - *this;
				return DirectionVec.Lenght();
			}

			// Calcualte direction to vector b
			MATHARZ_INLINE number_t Direction(const template_vec2<number_t> b) const
			{
				template_vec2<number_t> DirectionVec = b - *this;
				return DirectionVec;
			}

			// Calcualte normalized direction to vector b
			MATHARZ_INLINE number_t DirectionNormalized(const template_vec2<number_t> b) const
			{
				template_vec2<number_t> DirectionVec = b - *this;
				return DirectionVec.Normalize();
			}

			// Scalar addition
			// @return Vector with results of addition of each element with scalar 
			MATHARZ_INLINE template_vec2<number_t> ScalarAdd(const number_t b) const
			{
				return { x + b, y + b };
			};

			// Addition
			// @return Vector with sum of each corresponding elements
			MATHARZ_INLINE template_vec2<number_t> operator +(template_vec2<number_t> vec2) const {
				return { data[0] + vec2.data[0] , data[1] + vec2.data[1] };
			};

			// Scalar addition
			// @return Vector with results of addition of each element with scalar 
			MATHARZ_INLINE template_vec2<number_t> operator+(const number_t b) const
			{
				return ScalarAdd(b);
			};

			// Addition
			// @return This vector with sum of each corresponding elements of vectors
			MATHARZ_INLINE template_vec2<number_t>& operator+=(const template_vec2<number_t> b)
			{
				x += b.x;
				y += b.y;
				return *this;
			};

			// Scalar addition
			// @return This vector with sum of each corresponding elements with scalar
			MATHARZ_INLINE template_vec2<number_t>& operator+=(const number_t b)
			{
				x += b;
				y += b;
				return *this;
			};

			// -

			// Subtraction
			// @return Vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec2<number_t> operator-(const template_vec2<number_t> b) const
			{
				return { x - b.x, y - b.y };
			};

			// Subtraction
			// @return This vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec2<number_t>& operator-=(const template_vec2<number_t> b) const
			{
				x -= b.x;
				y -= b.y;
				return *this;
			};

			// scalar -

			// Scalar subtraction
			// @return Vector with difference of each element with scalar
			MATHARZ_INLINE template_vec2<number_t> operator-(const number_t b) const
			{
				return { x - b, y - b };
			};

			// Scalar subtraction
			// @return This vector with difference of each element with scalar
			MATHARZ_INLINE template_vec2<number_t>& operator-=(const number_t b)
			{
				x -= b;
				y -= b;
				return *this;
			};

			// *

			// Multiplication
			// @return Vector with results of multiplication of each elements of vector
			MATHARZ_INLINE template_vec2<number_t> operator*(const template_vec2<number_t> b) const
			{
				return { x * b.x, y * b.y };
			};

			// Multiplication
			// @return This vector with results of multiplication of each elements of vector
			MATHARZ_INLINE template_vec2<number_t>& operator*=(const template_vec2<number_t> b)
			{
				x *= b.x;
				y *= b.y;
				return *this;
			};

			// scalar *

			// Scalar multiplication
			// @return Vector with results of multiplication of each element with scalar
			MATHARZ_INLINE template_vec2<number_t> operator*(const number_t b) const
			{
				return { x * b, y * b };
			};

			// Scalar multiplication
			// @return This vector with results of multiplication of each element with scalar
			MATHARZ_INLINE template_vec2<number_t>& operator*=(const number_t b)
			{
				x *= b;
				y *= b;
				return *this;
			};

			// /

			// Division
			// @return Vector with results of division of each elements of vectors 
			MATHARZ_INLINE template_vec2<number_t> operator/(const template_vec2<number_t> b) const
			{
				MATHARZ_ASSERT((b.x != 0 && b.y != 0), "vec2 Divide by ZERO!")

				return { x / b.x, y / b.y };
			};

			// Division
			// @return This vector with results of division of each elements of vectors 
			MATHARZ_INLINE template_vec2<number_t>& operator/=(const template_vec2<number_t> b)
			{
				MATHARZ_ASSERT((b.x != 0 && b.y != 0), "vec2 Divide by ZERO!")
				x /= b.x;
				y /= b.y;
				return *this;
			};

			// scalar /

			// Scalar division
			// @return Vector with results of division of each element on scalar 
			MATHARZ_INLINE template_vec2<number_t> operator/(const number_t b) const
			{
				MATHARZ_ASSERT((b != 0), "vec2 Divide by ZERO!")

				return { x / b, y / b };
			};

			// Scalar division
			// @return This vector with results of division of each element on scalar 
			MATHARZ_INLINE template_vec2<number_t>& operator/=(const number_t b)
			{
				MATHARZ_ASSERT((b != 0), "vec2 Divide by ZERO!")
				x /= b;
				y /= b;
				return *this;
			};

			// Bool Compare overloads

			MATHARZ_INLINE bool operator ==(const template_vec2<number_t> vec2) const {
				return { equal(data[0] , vec2.data[0]) && equal(data[1] , vec2.data[1]) };
			};

			MATHARZ_INLINE bool operator >=(template_vec2<number_t> vec2) const {
				return  more_or_equal(data[0], vec2.data[0]) && (more_or_equal(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator <=(template_vec2<number_t> vec2) const {
				return  less_or_equal(data[0], vec2.data[0]) && (less_or_equal(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator >(template_vec2<number_t> vec2) const {
				return  more(data[0], vec2.data[0]) && (more(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator <(template_vec2<number_t> vec2) const {
				return  less(data[0], vec2.data[0]) && (less(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator ==(const number_t scalar) const {
				return { equal(data[0] , scalar) && equal(data[1] , scalar) };
			};

			MATHARZ_INLINE bool operator >=(number_t scalar) const {
				return  more_or_equal(data[0], scalar) && (more_or_equal(data[1], scalar));
			};

			MATHARZ_INLINE bool operator <=(number_t scalar) const {
				return  less_or_equal(data[0], scalar) && (less_or_equal(data[1], scalar));
			};

			MATHARZ_INLINE bool operator >(number_t scalar) const {
				return  more(data[0], scalar) && (more(data[1], scalar));
			};

			MATHARZ_INLINE bool operator <(number_t scalar) const {
				return  less(data[0], scalar) && (less(data[1], scalar));
			};

			// Access operators
			MATHARZ_INLINE number_t& operator[](size_t i) { MATHARZ_ASSERT((i < 2), "Out of size index"); return data[i]; };
			MATHARZ_INLINE const number_t& operator[](size_t i) const { MATHARZ_ASSERT((i < 2), "Out of size index"); return data[i]; };
		}; // template vec2

		// Vector with 3 components
		template<typename number_t>
		struct template_vec3 {
		public:
			union {
				number_t data[3];
				number_t rowData[1][3];
				number_t columnData[3][1];
				number_t lineArrayData[3];
				struct { number_t x, y, z; };
				struct { number_t r, g, b; };
			};

			template_vec3<number_t>() { std::fill_n(&lineArrayData[0], 3, 0); };

			template_vec3<number_t>(number_t Scalar) { std::fill_n(&lineArrayData[0], 3, Scalar); };

			template_vec3<number_t>(number_t x, number_t y, number_t z)
				: x(x), y(y), z(z) {};

			explicit template_vec3<number_t>(number_t data[3])
				: x(data[0]), y(data[1]), z(data[2]) {};

			template_vec3<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 3, "Init list doesn't a proper size(>3 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 3, *initList.begin());
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 3), data);
				};
			}

			explicit template_vec3<number_t>(template_vec2<number_t> vec2, number_t z)
				: x(vec2.x), y(vec2.y), z(z) {};

			explicit template_vec3<number_t>(number_t x, template_vec2<number_t> vec2)
				: x(x), y(vec2.x), z(vec2.y) {};

			// Cast to another vec_3 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_vec3< different_number_t>() const
			{
				return { static_cast<different_number_t>(this->x),static_cast<different_number_t>(this->y),static_cast<different_number_t>(this->z) };
			};

			MATHARZ_INLINE bool NearZero() const
			{
				return less(abs(*this), MTHRZ_FLOAT_EPSILON);
			}
			// Scalar addition
			// @return Vector with results of addition of each element with scalar 
			MATHARZ_INLINE template_vec3<number_t> ScalarAdd(const number_t b) const
			{
				return { x + b, y + b, z + b };
			}

			// Addition
			// @return Vector with results of addition of each elements of vectors
			MATHARZ_INLINE template_vec3<number_t> operator +(template_vec3<number_t> vec3) const {
				return { data[0] + vec3.data[0] , data[1] + vec3.data[1] , data[2] + vec3.data[2] };
			};

			// Scalar addition
			// @return Vector with results of addition of each element with scalar 
			MATHARZ_INLINE template_vec3<number_t> operator+(const number_t b) const
			{
				return ScalarAdd(b);
			}

			// Addition
			// @return This vector with results of addition of each elements of vectors
			MATHARZ_INLINE template_vec3<number_t>& operator+=(const template_vec3<number_t> b)
			{
				x += b.x;
				y += b.y;
				z += b.z;
				return *this;
			}

			// Scalar addition
			// @return This vector with results of addition of each elements of vector with scalar
			MATHARZ_INLINE template_vec3<number_t>& operator+=(const number_t b)
			{
				x += b;
				y += b;
				z += b;
				return *this;
			}

			// -

			// Subtraction
			// @return Vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec3<number_t> operator-(const template_vec3<number_t> b) const
			{
				return { x - b.x, y - b.y, z - b.z };
			}

			// Subtraction
			// @return This vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec3<number_t>& operator-=(const template_vec3<number_t> b)
			{
				x -= b.x;
				y -= b.y;
				z -= b.z;
				return *this;
			}
			// scalar -

			// Scalar subtraction
			// @return Vector with difference of each element with scalar
			MATHARZ_INLINE template_vec3<number_t> ScalarSubtract(const number_t b) const
			{
				return { x - b, y - b, z - b };
			}

			// Scalar subtraction
			// @return Vector with difference of each element with scalar
			MATHARZ_INLINE template_vec3<number_t> operator-(const number_t b) const
			{
				return ScalarSubtract(b);
			}

			// Scalar subtraction
			// @return This vector with difference of each element with scalar
			MATHARZ_INLINE template_vec3<number_t>& operator-=(const number_t b)
			{
				x -= b;
				y -= b;
				z -= b;
				return *this;
			}

			// *

			// Multiplication
			// @return Vector with results of multiplication of each elements of vector
			MATHARZ_INLINE template_vec3<number_t> operator*(const template_vec3<number_t> b) const
			{
				return { x * b.x, y * b.y, z * b.z };
			}

			// Multiplication
			// @return This vector with results of multiplication of each elements of vector
			MATHARZ_INLINE template_vec3<number_t>& operator*=(const template_vec3<number_t> b)
			{
				x *= b.x;
				y *= b.y;
				z *= b.z;
				return *this;
			}

			// scalar *

			// Scalar multiplication
			// @return Vector with results of multiplication of each element on scalar
			MATHARZ_INLINE template_vec3<number_t> ScalarMultiply(const number_t b) const
			{
				return { x * b, y * b, z * b };
			}

			// Scalar multiplication
			// @return Vector with results of multiplication of each element on scalar
			MATHARZ_INLINE template_vec3<number_t> operator*(const number_t b) const
			{
				return ScalarMultiply(b);
			}

			// Scalar multiplication
			// @return This vector with results of multiplication of each element on scalar
			MATHARZ_INLINE template_vec3<number_t>& operator*=(const number_t b)
			{
				x *= b;
				y *= b;
				z *= b;
				return *this;
			}

			// /

			// Division
			// @return Vector with results of division of each elements of vectors 
			MATHARZ_INLINE template_vec3<number_t> operator/(const template_vec3<number_t> b) const
			{
				MATHARZ_ASSERT((b.x != 0 && b.y != 0 && b.z != 0), "VEC3 Divide by ZERO!")

				return { x / b.x, y / b.y, z / b.z };
			};

			// Division
			// @return This vector with results of division of each elements of vectors 
			MATHARZ_INLINE template_vec3<number_t>& operator/=(const template_vec3<number_t> b)
			{
				MATHARZ_ASSERT((b.x != 0 && b.y != 0 && b.z != 0), "VEC3 Divide by ZERO!")
				x /= b.x;
				y /= b.y;
				z /= b.z;
				return *this;
			};

			// scalar /

			// Scalar division
			// @return Vector with results of division of each element with scalar 
			MATHARZ_INLINE template_vec3<number_t> ScalarDivide(const number_t b) const
			{
				MATHARZ_ASSERT((b != 0), "VEC3 Divide by ZERO!")
				return { x / b, y / b, z / b };
			}

			// Scalar division
			// @return Vector with results of division of each element with scalar 
			MATHARZ_INLINE template_vec3<number_t> operator/(const number_t b) const
			{
				MATHARZ_ASSERT((b != 0), "VEC3 Divide by ZERO!")

				return ScalarDivide(b);
			}

			// Scalar division 
			// @return This vector with results of division of each element with scalar 
			MATHARZ_INLINE template_vec3<number_t>& operator/=(const number_t b)
			{
				MATHARZ_ASSERT((b != 0), "VEC3 Divide by ZERO!")
				x /= b;
				y /= b;
				z /= b;
				return *this;
			}

			// Math functions

			MATHARZ_INLINE number_t SquareLength() const
			{
				return { x * x + y * y + z * z };
			}

			MATHARZ_INLINE number_t SquareMagnitude() const
			{
				return SquareLength();
			}

			MATHARZ_INLINE number_t SumOfElements() const
			{
				return x + y + z;
			}

			MATHARZ_INLINE number_t Lenght() const
			{
				return static_cast<number_t>(sqrt(SquareLength()));
			}

			MATHARZ_INLINE number_t Magnitude() const
			{
				return Lenght();
			}

			MATHARZ_INLINE number_t DotProductFromSelf() const
			{
				return SquareLength();
			}

			// Calculate dotProduct
			// @return dotProduct
			MATHARZ_INLINE number_t DotProduct(const template_vec3<number_t> b) const
			{
				return { x * b.x + y * b.y + z * b.z };
			}

			// Calculate crossProduct
			// @return vector with result crossProduct
			MATHARZ_INLINE template_vec3<number_t> CrossProduct(const template_vec3<number_t> b) const
			{
				template_vec3<number_t> result;
				result.data[0] = this->data[1] * b.data[2] - this->data[2] * b.data[1];
				result.data[1] = this->data[2] * b.data[0] - this->data[0] * b.data[2];
				result.data[2] = this->data[0] * b.data[1] - this->data[1] * b.data[0];
				return std::move(result);
			}

			// Get inverse from this vector
			// @return Inversed vector
			MATHARZ_INLINE template_vec3<number_t> GetInversed() const
			{
				return { x * -1,y * -1,z * -1 };
			}

			// Inverse this vector
			// @return Ref to this (inversed)vector
			MATHARZ_INLINE template_vec3<number_t>& Inverse()
			{
				x *= -1;
				y *= -1;
				z *= -1;
				return *this;
			}

			// Get normalized vector
			MATHARZ_INLINE template_vec3<number_t> GetNormalized() const
			{
				number_t lenght = Lenght();
				MATHARZ_ASSERT(!(equal(lenght,0)), "VEC3 Divide by ZERO!")
				return { x / lenght,y / lenght,z / lenght };
			}

			// Normalize this vector
			MATHARZ_INLINE template_vec3<number_t>& Normalize()
			{
				number_t lenght = Lenght();
				MATHARZ_ASSERT(!(equal(lenght,0)), "VEC3 Divide by ZERO!")
				x /= lenght;
				y /= lenght;
				z /= lenght;
				return *this;
			}

			// Calcualte distance to vector b
			MATHARZ_INLINE number_t DistanceTo(const template_vec3<number_t> b) const
			{
				template_vec3<number_t> DirectionVec = b - *this;
				return DirectionVec.Lenght();
			}

			// Calcualte direction to vector b
			MATHARZ_INLINE number_t Direction(const template_vec3<number_t> b) const
			{
				template_vec3<number_t> DirectionVec = b - *this;
				return DirectionVec;
			}

			// Calcualte normalized direction to vector b
			MATHARZ_INLINE number_t DirectionNormalized(const template_vec3<number_t> b) const
			{
				template_vec3<number_t> DirectionVec = b - *this;
				return DirectionVec.Normalize();
			}

			// Check if this vector is perpendicular to vector b
			MATHARZ_INLINE bool IsPerpendicularTo_INT(const template_vec3<number_t> b) const
			{
				return (equal(DotProduct(b), static_cast<number_t>(0)));
			}

			// Check if this vector is parallel to vector b
			MATHARZ_INLINE bool IsParallelTo_INT(const template_vec3<number_t> b) const
			{
				return (equal(CrossProduct(b), static_cast<number_t>(0)));
			}

			// Check if this vector is perpendicular to vector b with safe guard
			MATHARZ_INLINE bool IsPerpendicularTo(const template_vec3<number_t> b, number_t safeGuard = 0.15) const
			{
				auto tempDP = DotProduct(b);
				return less_or_equal(abs(tempDP), safeGuard);
			}

			/*	Check if this vector is parallel to vector b with safe guard
				(for avoiding bugs with floating point, (false)(0.0001 == 0) ; (true)(-safeGuard(.15) <0.0001 < safeGuard))
			*/
			MATHARZ_INLINE bool IsParallelTo(const template_vec3<number_t> b, number_t safeGuard = 0.15) const
			{
				auto tempVector = CrossProduct(b);
				return less_or_equal(abs(tempVector), safeGuard);
			}

			// Pow each element on xE value
			MATHARZ_INLINE template_vec3<number_t> PowEachElement(const number_t xE) const
			{
				return { pow(x,xE),pow(y,xE),pow(z,xE) };
			}

			// Pow each element on corresponding b value axis
			MATHARZ_INLINE template_vec3<number_t> PowEachElement(const template_vec3<number_t> b) const
			{
				return { pow(x,b.x),pow(y,b.y),pow(z,b.z) };
			}

			// Pow each element on xE value
			MATHARZ_INLINE template_vec3<number_t>& SelfPowEachElement(const number_t xE)
			{
				x = pow(x, xE);
				y = pow(y, xE);
				z = pow(z, xE);
				return *this;
			};

			// Pow each element on corresponding b value axis
			MATHARZ_INLINE template_vec3<number_t>& SelfPowEachElement(const template_vec3<number_t> b)
			{
				x = pow(x, b.x);
				y = pow(y, b.y);
				z = pow(z, b.z);
				return *this;
			};

			// Projection from b vector onto this
			MATHARZ_INLINE template_vec3<number_t> ProjectionFrom(const template_vec3<number_t> b) const
			{
				MATHARZ_ASSERT((SquareLength() != 0), "VEC3 Divide by ZERO!")
					return (*this * (b.DotProduct(*this) / SquareLength()));
			}

			// Projection on b vector from this
			MATHARZ_INLINE template_vec3<number_t> Projection(const template_vec3<number_t> b) const
			{
				return b.ProjectionFrom(*this);
			}

			// Get angle(in radians) between this vector and vector b
			MATHARZ_INLINE double Angle(const template_vec3<number_t>& b)
			{
				double DotPrdct = GetNormalized().DotProduct(b.GetNormalized());

				if (more_or_equal(DotPrdct, 1.0))
					return 0.0;
				else if (less_or_equal(DotPrdct, -1.0))
					return pi<double>();
				else
					return math::acos(DotPrdct);
			}

			// RAY TRACING MATH
			// -----------------------
			// Calculate reflected ray direction from surface with some normal(n), reflectFactor - to negate vector in-surface part
			MATHARZ_INLINE template_vec3<number_t> Reflect(const template_vec3<number_t>& n) {
				const number_t reflectFactor = 2;
				return *this - (n * reflectFactor * DotProduct(n));
			}

			MATHARZ_INLINE template_vec3<number_t> Refract(const template_vec3<number_t>& n, double etai_over_etat)
			{
				double cosTheta = min((*this * -1.f).DotProduct(n), 1.0);

				template_vec3<number_t> rOutPerp = (*this + n * cosTheta) * etai_over_etat;
				template_vec3<number_t> rOutParallel = n * -sqrt(abs(1.0 - rOutPerp.SquareLength()));
				return rOutPerp + rOutParallel;
			}

			//-------------------------

			// overloaded XOR operator for use as Pow operation(on each element)

			MATHARZ_INLINE template_vec3<number_t> operator^(const template_vec3<number_t> b) const
			{
				return PowEachElement(b);
			}

			MATHARZ_INLINE template_vec3<number_t>& operator^=(const template_vec3<number_t> p)
			{
				return SelfPowEachElement(p);
			}

			MATHARZ_INLINE template_vec3<number_t> operator^(const number_t p) const
			{
				return PowEachElement(p);
			}

			MATHARZ_INLINE template_vec3<number_t>& operator^=(const number_t p)
			{
				return SelfPowEachElement(p);
			}


			// bool operations

			MATHARZ_INLINE bool operator !=(number_t number) const {
				return !(equal(data[0], number));
			};

			MATHARZ_INLINE bool operator !=(template_vec2<number_t> vec2) const {
				return !(equal(data[0], vec2.data[0]) && equal(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator !=(template_vec3<number_t> vec3) const {
				return !(equal(data[0], vec3.data[0]) && equal(data[1], vec3.data[1]) && equal(data[2], vec3.data[2]));
			};

			MATHARZ_INLINE bool operator ==(number_t number) const {
				return (equal(data[0], number));
			};

			MATHARZ_INLINE bool operator ==(template_vec2<number_t> vec2) const {
				return (equal(data[0], vec2.data[0]) && equal(data[1], vec2.data[1]));
			};

			MATHARZ_INLINE bool operator ==(template_vec3<number_t> vec3) const {
				return (equal(*this, vec3));
			};

			MATHARZ_INLINE bool operator >=(number_t number) const {
				return { more_or_equal(data[0],number) };
			};

			MATHARZ_INLINE bool operator >=(template_vec2<number_t> vec2) const {
				return { more_or_equal(data[0] , vec2.data[0]) && (more_or_equal(data[1] ,vec2.data[1])) };
			};

			MATHARZ_INLINE bool operator >=(template_vec3<number_t> vec3) const {
				return { more_or_equal(data[0] , vec3.data[0]) && (more_or_equal(data[1] ,vec3.data[1])) && more_or_equal(data[2] , vec3.data[2]) };
			};

			MATHARZ_INLINE bool operator <=(number_t number) const {
				return { less_or_equal(data[0], number) && less_or_equal(data[1], number)  && less_or_equal(data[2], number) };
			};

			MATHARZ_INLINE bool operator <=(template_vec2<number_t> vec2) const {
				return { less_or_equal(data[0] , vec2.data[0]) && less_or_equal(data[1] , vec2.data[1]) };
			}

			MATHARZ_INLINE bool operator <=(template_vec3<number_t> vec3) const {
				return { less_or_equal(data[0], vec3.data[0]) && less_or_equal(data[1] ,vec3.data[1]) && less_or_equal(data[2] , vec3.data[2]) };
			};

			MATHARZ_INLINE bool operator >(number_t number) const {
				return { more(data[0] , number) };
			};

			MATHARZ_INLINE bool operator >(template_vec2<number_t> vec2) const {
				return { more(data[0], vec2.data[0]) && more(data[1] , vec2.data[1]) };
			};

			MATHARZ_INLINE bool operator >(template_vec3<number_t> vec3) const {
				return { more(data[0] , vec3.data[0]) && more(data[1] ,vec3.data[1]) && more(data[2] , vec3.data[2]) };
			};

			MATHARZ_INLINE bool operator <(number_t number) const {
				return { less(data[0] , number) };
			};

			MATHARZ_INLINE bool operator <(template_vec2<number_t> vec2) const {
				return { less(data[0] , vec2.data[0]) && less(data[1] ,vec2.data[1]) };
			};

			MATHARZ_INLINE bool operator <(template_vec3<number_t> vec3) const {
				return { less(data[0] ,vec3.data[0]) && less(data[1] ,vec3.data[1]) && less(data[2] ,vec3.data[2]) };
			};

			MATHARZ_INLINE bool operator !=(template_vec4<number_t> vec4) const {
				return !(equal(vec4.xyz, *this));
			};

			MATHARZ_INLINE bool operator ==(template_vec4<number_t> vec4) const {
				return equal(vec4.xyz, *this);
			};

			MATHARZ_INLINE bool operator >=(template_vec4<number_t> vec4) const {
				return more_or_equal(vec4.xyz, *this);
			};

			MATHARZ_INLINE bool operator <=(template_vec4<number_t> vec4) const {
				return less_or_equal(vec4.xyz, *this);
			};

			MATHARZ_INLINE bool operator >(template_vec4<number_t> vec4) const {
				return more(vec4.xyz, *this);
			};

			MATHARZ_INLINE bool operator <(template_vec4<number_t> vec4) const {
				return less(vec4.xyz, *this);
			};

			// Access operators
			MATHARZ_INLINE number_t& operator[](size_t i) { MATHARZ_ASSERT((i < 3), "Out of size index"); return data[i]; };
			MATHARZ_INLINE const number_t& operator[](size_t i) const { MATHARZ_ASSERT((i < 3), "Out of size index"); return data[i]; };
		}; // template_vec3

		template<typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator*(number_t a, const template_vec3<number_t>& vec)
		{
			return vec.ScalarMultiply(a);
		}

		template<typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator/(number_t a, const template_vec3<number_t>& vec)
		{
			return vec.ScalarDivide(a);
		}
		template<typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator+(number_t a, const template_vec3<number_t>& vec)
		{
			return vec.ScalarAdd(a);
		}
		template<typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator-(number_t a, const template_vec3<number_t>& vec)
		{
			return vec.ScalarSubtract(a);
		}

		// Get angle(in radians) between two vectors
		template<typename number_t>
		MATHARZ_INLINE double Angle(const template_vec3<number_t>& a, const template_vec3<number_t>& b)
		{
			return a.Angle(b);
		}

		// Vector with 4 components
		template<typename number_t>
		struct template_vec4 {
		public:
			union {
				number_t data[4];
				number_t rowData[1][4];
				number_t columnData[4][1];
				number_t lineArrayData[4];
				struct { number_t x, y, z, w; };
				struct { number_t r, g, b, a; };
				struct { template_vec3<number_t> xyz; number_t IgnoredW; };
				struct { number_t IgnoredX; template_vec3<number_t> yzw; };
				struct { template_vec3<number_t> rgb; number_t IgnoredA; };
				struct { number_t IgnoredR; template_vec3<number_t> gba; };
				struct { template_vec2<number_t> xy; template_vec2<number_t> zw; };
				struct { template_vec2<number_t> rg; template_vec2<number_t> ba; };
			};

			template_vec4<number_t>() { std::fill_n(&lineArrayData[0], 4, 0); };

			template_vec4<number_t>(number_t x, number_t y, number_t z = 0, number_t w = 0)
				: x(x), y(y), z(z), w(w) {};

			template_vec4<number_t>(number_t scalar)
				: x(scalar), y(scalar), z(scalar), w(scalar) {};

			template <typename other_number_t>
			template_vec4<number_t>(template_quaternion< other_number_t> quaternion)
			{
				auto castedQuat = static_cast<template_quaternion<number_t>>(quaternion);
				std::copy_n(&castedQuat.lineArrayData[0], 4, &data[0]);
			};

			template_vec4<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 4, "Init list doesn't a proper size(>4 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 4, *initList.begin());
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 4), data);
				};
			}

			explicit template_vec4<number_t>(number_t xyzw[4])
			{
				std::copy_n(&xyzw[0], 4, &data[0]);
			};

			explicit template_vec4<number_t>(template_vec2<number_t> xyVec, template_vec2<number_t> zwVec)
				: xy(xyVec), zw(zwVec) {};

			explicit template_vec4<number_t>(template_vec3<number_t> vec3, number_t w)
				: xyz(vec3), w(w) {};

			explicit template_vec4<number_t>(number_t x, template_vec3<number_t> vec3)
				: x(x), yzw(vec3) {};

			// Cast to another vec_4 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_vec4< different_number_t>() const
			{
				return { static_cast<different_number_t>(this->x),static_cast<different_number_t>(this->y),
					static_cast<different_number_t>(this->z),static_cast<different_number_t>(this->w) };
			};

			MATHARZ_INLINE bool NearZero() const
			{
				return less(abs(*this), MTHRZ_FLOAT_EPSILON);
			}
			// Scalar multiplication
			// @return Vector with results of multiplication of each element on scalar
			MATHARZ_INLINE template_vec4<number_t> ScalarMultiply(const number_t b) const
			{
				return template_vec4<number_t>{ x * b, y * b, z * b, w * b };
			}

			// Scalar multiplication
			// @return Vector with results of multiplication of each element on scalar
			MATHARZ_INLINE template_vec4<number_t> operator*(const number_t b) const
			{
				return ScalarMultiply(b);
			}

			// Vector multiplication
			// @return Vector with results of multiplication of each corresponding elements of vectors
			MATHARZ_INLINE template_vec4<number_t> Multiply(const template_vec4<number_t> b) const
			{
				return template_vec4<number_t> { x * b.x, y * b.y, z * b.z, w * b.w };
			}

			// Vector multiplication
			// @return Vector with results of multiplication of each corresponding elements of vectors
			MATHARZ_INLINE template_vec4<number_t> operator*(const template_vec4<number_t> b) const
			{
				return Multiply(b);
			}

			// Subtraction
			// @return Vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec4<number_t> operator-(const template_vec4<number_t> b) const
			{
				return template_vec4<number_t> { x - b.x, y - b.y, z - b.z, w - b.w };
			}

			// Subtraction
			// @return This vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec4<number_t>& operator-=(const template_vec4<number_t> b) const
			{
				x -= b.x;
				y -= b.y;
				z -= b.z;
				w -= b.w;
				return *this;
			}

			// Addition
			// @return Vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec4<number_t> operator+(const template_vec4<number_t> b) const
			{
				return template_vec4<number_t> { x + b.x, y + b.y, z + b.z, w + b.w };
			}

			// Addition
			// @return This vector with difference of each elements of vectors
			MATHARZ_INLINE template_vec4<number_t>& operator+=(const template_vec4<number_t> b) const
			{
				x += b.x;
				y += b.y;
				z += b.z;
				w += b.w;
				return *this;
			}

			// bool operators
			MATHARZ_INLINE bool operator !=(template_vec4<number_t> vec4) const {
				return !(equal(data[0], vec4.data[0]) && equal(data[1], vec4.data[1])
					&& equal(data[2], vec4.data[2]) && equal(data[3], vec4.data[3]));
			};

			MATHARZ_INLINE bool operator ==(template_vec4<number_t> vec4) const {
				return equal(data[0], vec4.data[0]) && equal(data[1], vec4.data[1])
					&& equal(data[2], vec4.data[2]) && equal(data[3], vec4.data[3]);
			};

			MATHARZ_INLINE bool operator >=(template_vec4<number_t> vec4) const {
				return  more_or_equal(data[0], vec4.data[0]) && (more_or_equal(data[1], vec4.data[1]))
					&& more_or_equal(data[2], vec4.data[2]) && more_or_equal(data[3], vec4.data[3]);
			};

			MATHARZ_INLINE bool operator <=(template_vec4<number_t> vec4) const {
				return  less_or_equal(data[0], vec4.data[0]) && (less_or_equal(data[1], vec4.data[1]))
					&& less_or_equal(data[2], vec4.data[2]) && less_or_equal(data[3], vec4.data[3]);
			};

			MATHARZ_INLINE bool operator >(template_vec4<number_t> vec4) const {
				return more(data[0], vec4.data[0]) && (more(data[1], vec4.data[1]))
					&& more(data[2], vec4.data[2]) && more(data[3], vec4.data[3]);
			};

			MATHARZ_INLINE bool operator <(template_vec4<number_t> vec4) const {
				return  less(data[0], vec4.data[0]) && (less(data[1], vec4.data[1]))
					&& less(data[2], vec4.data[2]) && less(data[3], vec4.data[3]);
			};

			// bool operators
			MATHARZ_INLINE bool operator !=(template_vec3<number_t> vec3) const {
				return !(xyz == vec3);
			};

			MATHARZ_INLINE bool operator ==(template_vec3<number_t> vec3) const {
				return xyz == vec3;
			};

			MATHARZ_INLINE bool operator >=(template_vec3<number_t> vec3) const {
				return xyz >= vec3;
			};

			MATHARZ_INLINE bool operator <=(template_vec3<number_t> vec3) const {
				return xyz <= vec3;
			};

			MATHARZ_INLINE bool operator >(template_vec3<number_t> vec3) const {
				return xyz > vec3;
			};

			MATHARZ_INLINE bool operator <(template_vec3<number_t> vec3) const {
				return xyz < vec3;
			};

			// Access operators
			MATHARZ_INLINE number_t& operator[](size_t i) {
				MATHARZ_ASSERT((i < 4), "Out of size index"); return data[i];
			};
			MATHARZ_INLINE const number_t& operator[](size_t i) const { MATHARZ_ASSERT((i < 4), "Out of size index"); return data[i]; };
		}; // template_vec4

		// Quaternion
		template<typename number_t>
		struct template_quaternion {
		public:
			union
			{
				alignas(sizeof(number_t) * 4) number_t data[4];
				alignas(sizeof(number_t) * 4) number_t rowData[1][4];
				alignas(sizeof(number_t) * 4) number_t columnData[4][1];
				alignas(sizeof(number_t) * 4) number_t lineArrayData[4];
				struct { alignas(sizeof(number_t) * 4) template_vec4<number_t> vector; };
				struct { number_t a, b, c, d; };
				struct { number_t x, y, z, w; };
				struct { alignas(sizeof(number_t) * 4) template_vec3<number_t> xyz; number_t IgnoredW; };
				struct { number_t IgnoredX; alignas(sizeof(number_t) * 4) template_vec3<number_t> yzw; };
				struct { alignas(sizeof(number_t) * 2) template_vec2<number_t> xy; alignas(sizeof(number_t) * 2) template_vec2<number_t> zw; };
			};

			template_quaternion<number_t>() { std::fill_n(&lineArrayData[0], 4, 0); };

			// Construct quaternion from x,y,z,w values
			template_quaternion<number_t>(number_t x, number_t y, number_t z, number_t w)
				: x(x), y(y), z(z), w(w) {};

			// Construct quaternion from euler angles(radians)(similar to call MakeFromEulerRotation on quaternion).
			template_quaternion<number_t>(number_t x, number_t y, number_t z)
			{
				MakeFromEulerRotation(x, y, z);
			};

			// Construct quaternion from euler angles(radians)(similar to call MakeFromEulerRotation on quaternion).
			template_quaternion<number_t>(template_vec3<number_t> vec3)
			{
				MakeFromEulerRotation(vec3);
			};

			template_quaternion<number_t>(number_t scalar)
				: x(scalar), y(scalar), z(scalar), w(scalar) {};

			template_quaternion<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 4, "Init list doesn't a proper size(>4 numbers)!");

				std::copy_n(initList.begin(), clamp((int)(initList.end() - initList.begin()), 0, 4), data);
			}

			explicit template_quaternion<number_t>(number_t xyzw[4])
			{
				std::copy_n(&xyzw[0], 4, &data[0]);
			};

			explicit template_quaternion<number_t>(template_vec2<number_t> xyVec, template_vec2<number_t> zwVec)
				: xy(xyVec), zw(zwVec) {};

			// Construct from vec3 and w value as from axis angle(similar to call MakeFromAxisAngle on quaternion)
			explicit template_quaternion<number_t>(template_vec3<number_t> vec3, number_t w)
			{
				MakeFromAxisAngle(vec3, w);
			}

			// Cast to another vec_4 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_quaternion< different_number_t>() const
			{
				return { static_cast<different_number_t>(this->x),static_cast<different_number_t>(this->y),
					static_cast<different_number_t>(this->z),static_cast<different_number_t>(this->w) };
			};

			// Math functions

			// Multiply quaternions
			// @return Result quaternion
			MATHARZ_INLINE template_quaternion<number_t> Multiply(template_quaternion<number_t>  q_1) const {
				template_quaternion<number_t> out_quaternion{};

				out_quaternion.x = this->x * q_1.w +
					this->y * q_1.z -
					this->z * q_1.y +
					this->w * q_1.x;

				out_quaternion.y = -this->x * q_1.z +
					this->y * q_1.w +
					this->z * q_1.x +
					this->w * q_1.y;

				out_quaternion.z = this->x * q_1.y -
					this->y * q_1.x +
					this->z * q_1.w +
					this->w * q_1.z;

				out_quaternion.w = -this->x * q_1.x -
					this->y * q_1.y -
					this->z * q_1.z +
					this->w * q_1.w;

				return out_quaternion;
			}

			// Implementation source from Cry engine code
			// Multiply quaternion with vec3 (make 3d rotation)
			// @return Result vector
			MATHARZ_INLINE template_vec3<number_t> Multiply(const template_vec3<number_t>& v) {
				MATHARZ_CHECK((abs(1 - (DotProduct(*this)))) < 0.01); //check if unit-quaternion

				number_t vxvx = this->x * this->x;
				number_t vzvz = this->z * this->z;
				number_t vyvy = this->y * this->y;
				number_t vxvy = this->x * this->y;
				number_t vxvz = this->x * this->z;
				number_t vyvz = this->y * this->z;
				number_t svx = this->w * this->x, svy = this->w * this->y, svz = this->w * this->z;
				template_vec3<number_t> res{};
				res.x = v.x * (1 - (vyvy + vzvz) * 2) + v.y * (vxvy - svz) * 2 + v.z * (vxvz + svy) * 2;
				res.y = v.x * (vxvy + svz) * 2 + v.y * (1 - (vxvx + vzvz) * 2) + v.z * (vyvz - svx) * 2;
				res.z = v.x * (vxvz - svy) * 2 + v.y * (vyvz + svx) * 2 + v.z * (1 - (vxvx + vyvy) * 2);
				return std::move(res);
			}

			// Calculate dotProduct
			// @return DotProduct result
			MATHARZ_INLINE number_t DotProduct(template_quaternion<number_t>  q_1) const {
				return this->x * q_1.x +
					this->y * q_1.y +
					this->z * q_1.z +
					this->w * q_1.w;
			}

			// Calculate normal for given quaternion
			// @return Normal of given quaternion
			MATHARZ_INLINE number_t Normal() const {
				return matharz::sqrt(
					this->x * this->x +
					this->y * this->y +
					this->z * this->z +
					this->w * this->w);
			};

			// Normalize this quaternion
			// @return Normalized quaternion
			MATHARZ_INLINE template_quaternion<number_t>& Normalize() {
				number_t normal = Normal();
				this->x /= normal;
				this->y /= normal;
				this->z /= normal;
				this->w /= normal;
				return *this;
			};

			// Normalize quaternion
			// @return Normalized quaternion
			MATHARZ_INLINE template_quaternion< number_t> GetNormalized() const {
				number_t normal = Normal();
				return template_quaternion<number_t> {
					this->x / normal,
						this->y / normal,
						this->z / normal,
						this->w / normal
				};
			};

			// Conjugate quaternion
			// @return Conjugated quaternion
			MATHARZ_INLINE template_quaternion<number_t> GetConjugate() const {
				return template_quaternion< number_t>{
					-this->x,
						-this->y,
						-this->z,
						this->w
				};
			};

			// Conjugate quaternion
			// @return Conjugated quaternion
			MATHARZ_INLINE template_quaternion<number_t>& Conjugate() {
				this->x *= -1;
				this->y *= -1;
				this->z *= -1;
				return *this;
			};

			// Rotations and matrix stuff

			// Create a rotation matrix from quaternion
			// @return Rotation matrix
			// Source implementation from https://github.com/travisvroman/kohi/blob/main/engine/src/math/kmath.h
			MATHARZ_INLINE template_matrix4x4<number_t> ToRotationMatrix() const {
				template_matrix4x4<number_t> out_matrix{ 1.f };

				// https://stackoverflow.com/questions/1556260/convert-quaternion-rotation-to-rotation-matrix

				template_quaternion<number_t> n = GetNormalized();

				out_matrix.lineArrayData[0] = 1.0f - 2.0f * n.y * n.y - 2.0f * n.z * n.z;
				out_matrix.lineArrayData[1] = 2.0f * n.x * n.y - 2.0f * n.z * n.w;
				out_matrix.lineArrayData[2] = 2.0f * n.x * n.z + 2.0f * n.y * n.w;

				out_matrix.lineArrayData[4] = 2.0f * n.x * n.y + 2.0f * n.z * n.w;
				out_matrix.lineArrayData[5] = 1.0f - 2.0f * n.x * n.x - 2.0f * n.z * n.z;
				out_matrix.lineArrayData[6] = 2.0f * n.y * n.z - 2.0f * n.x * n.w;

				out_matrix.lineArrayData[8] = 2.0f * n.x * n.z - 2.0f * n.y * n.w;
				out_matrix.lineArrayData[9] = 2.0f * n.y * n.z + 2.0f * n.x * n.w;
				out_matrix.lineArrayData[10] = 1.0f - 2.0f * n.x * n.x - 2.0f * n.y * n.y;

				return out_matrix;
			};

			// Cast to matrix 4x4(perform ToMatrix operation) template(rotation matrix), be careful about with data you cast, possible lose of data
			explicit operator template_matrix4x4<number_t>() const
			{
				return ToRotationMatrix();
			};

			// Calculates a rotation matrix based on the quaternion and the passed in center point.
			// @return Rotation matrix
			// Source implementation from https://github.com/travisvroman/kohi/blob/main/engine/src/math/kmath.h
			MATHARZ_INLINE template_matrix4x4<number_t> ToRotationMatrix(template_vec3<number_t> center) const {
				template_matrix4x4<number_t>  out_matrix{};

				out_matrix.lineArrayData[0] = (this->x * this->x) - (this->y * this->y) - (this->z * this->z) + (this->w * this->w);
				out_matrix.lineArrayData[1] = 2.0f * ((this->x * this->y) + (this->z * this->w));
				out_matrix.lineArrayData[2] = 2.0f * ((this->x * this->z) - (this->y * this->w));
				out_matrix.lineArrayData[3] = center.x - center.x * out_matrix.lineArrayData[0] - center.y * out_matrix.lineArrayData[1] - center.z * out_matrix.lineArrayData[2];

				out_matrix.lineArrayData[4] = 2.0f * ((this->x * this->y) - (this->z * this->w));
				out_matrix.lineArrayData[5] = -(this->x * this->x) + (this->y * this->y) - (this->z * this->z) + (this->w * this->w);
				out_matrix.lineArrayData[6] = 2.0f * ((this->y * this->z) + (this->x * this->w));
				out_matrix.lineArrayData[7] = center.y - center.x * out_matrix.lineArrayData[4] - center.y * out_matrix.lineArrayData[5] - center.z * out_matrix.lineArrayData[6];

				out_matrix.lineArrayData[8] = 2.0f * ((this->x * this->z) + (this->y * this->w));
				out_matrix.lineArrayData[9] = 2.0f * ((this->y * this->z) - (this->x * this->w));
				out_matrix.lineArrayData[10] = -(this->x * this->x) - (this->y * this->y) + (this->z * this->z) + (this->w * this->w);
				out_matrix.lineArrayData[11] = center.z - center.x * out_matrix.lineArrayData[8] - center.y * out_matrix.lineArrayData[9] - center.z * out_matrix.lineArrayData[10];

				out_matrix.lineArrayData[15] = 1.0f;
				return out_matrix;
			}

			// Make this quaternion from the given axis and angle.
			template<typename angle_t = float>
			MATHARZ_INLINE template_quaternion<number_t>& MakeFromAxisAngle(template_vec3<number_t>  axis, angle_t angle) {
				template_vec3<number_t>   vn = axis.GetNormalized();
				angle = radians(angle);
				angle *= 0.5f;
				float sinAngle = sin(angle);

				template_quaternion<number_t> quat = template_quaternion<number_t>{ cos(angle), vn.x * sinAngle, vn.y * sinAngle, vn.z * sinAngle };
				std::swap(*this, quat);
				return *this;
			};

			// Make this quaternion from euler rotation
			template<typename angle_t = float>
			MATHARZ_INLINE template_quaternion<number_t>& MakeFromEulerRotation(angle_t X, angle_t Y, angle_t Z) {
				X = radians(x);
				Y = radians(y);
				Z = radians(z);

				X *= 0.5f;
				Y *= 0.5f;
				Z *= 0.5f;

				angle_t sinx = sin(X);
				angle_t siny = sin(Y);
				angle_t sinz = sin(Z);
				angle_t cosx = cos(X);
				angle_t cosy = cos(Y);
				angle_t cosz = cos(Z);

				this->w = cosx * cosy * cosz + sinx * siny * sinz;
				this->x = sinx * cosy * cosz + cosx * siny * sinz;
				this->y = cosx * siny * cosz - sinx * cosy * sinz;
				this->z = cosx * cosy * sinz - sinx * siny * cosz;
				return *this;
			}

			// Make this quaternion from euler rotation
			MATHARZ_INLINE template_quaternion<number_t>& MakeFromEulerRotation(template_vec3<number_t> vec) {
				return MakeFromEulerRotation(vec.x, vec.y, vec.z);
			}

			// Make liner interpolation between two quaterians
			// @return Interpolated quaternion
			// Source implementation from https://github.com/travisvroman/kohi/blob/main/engine/src/math/kmath.h
			template<typename float_t = float>
			template_quaternion<number_t> Slerp(template_quaternion<number_t> to, float_t threshold) {
				float_t cosTheta = this->DotProduct(to);
				template_quaternion<number_t> temp(to);

				if (cosTheta < 0.0f) {
					cosTheta *= -1.0f;
					temp = temp * -1.0f;
				}

				float_t theta = acos(cosTheta);
				float_t sinTheta = 1.0f / sin(theta);

				return sinTheta * (
					(template_quaternion<number_t>(*this * sin(theta * (1.0f - threshold)))) +
					(template_quaternion<number_t>(temp * sin(threshold * theta)))
					);
			};

			// Get angle axis from quaternion by given angle and vector reference
			template<typename angle_t = float>
			MATHARZ_INLINE void GetAngleAxis(angle_t& angle, template_vec3<number_t>& axis) const {
				angle = cos(w);
				angle_t sinz = sin(angle);

				if (abs(sinz) > MTHRZ_FLOAT_EPSILON) {
					sinz = 1.0f / sinz;

					axis->x = x * sinz;
					axis->y = y * sinz;
					axis->z = z * sinz;

					angle *= 2.0f * 57.2957795f;
					if (angle > 180.0f)
						angle = 360.0f - angle;
				}
				else {
					angle = 0.0f;
					axis->x = 1.0f;
					axis->y = 0.0f;
					axis->z = 0.0f;
				}
			}

			// Get Matrix
			MATHARZ_INLINE template_matrix4x4<number_t> GetMatrix() const
			{
				template_matrix4x4<number_t> resultMatrix{};
				resultMatrix.lineArrayData[0] = 1.0f - 2.0f * y * y - 2.0f * z * z;
				resultMatrix.lineArrayData[1] = 2.0f * x * y + 2.0f * z * w;
				resultMatrix.lineArrayData[2] = 2.0f * x * z - 2.0f * y * w;
				resultMatrix.lineArrayData[3] = 0.0f;

				resultMatrix.lineArrayData[4] = 2.0f * x * y - 2.0f * z * w;
				resultMatrix.lineArrayData[5] = 1.0f - 2.0f * x * x - 2.0f * z * z;
				resultMatrix.lineArrayData[6] = 2.0f * z * y + 2.0f * x * w;
				resultMatrix.lineArrayData[7] = 0.0f;

				resultMatrix.lineArrayData[8] = 2.0f * x * z + 2.0f * y * w;
				resultMatrix.lineArrayData[9] = 2.0f * z * y - 2.0f * x * w;
				resultMatrix.lineArrayData[10] = 1.0f - 2.0f * x * x - 2.0f * y * y;
				resultMatrix.lineArrayData[11] = 0.0f;
				return resultMatrix;
			};

			// Get Matrix with center vector
			MATHARZ_INLINE template_matrix4x4<number_t> GetMatrix(const template_vec3<number_t> centerVec3) const
			{
				auto resultMatrix = GetMatrix();

				resultMatrix.lineArrayData[12] = centerVec3.x;
				resultMatrix.lineArrayData[13] = centerVec3.y;
				resultMatrix.lineArrayData[14] = centerVec3.z;
				resultMatrix.lineArrayData[15] = 1.f;
				return resultMatrix;
			};

			// Math operators

			MATHARZ_INLINE template_quaternion<number_t> friend operator * (const template_quaternion<number_t> q, number_t t) {
				return template_quaternion<number_t>(q.xyz * t, q.w * t);
			};

			MATHARZ_INLINE template_quaternion<number_t> friend operator *(number_t t, const template_quaternion<number_t> q) {
				return template_quaternion<number_t>(q.xyz * t, q.w * t);
			};

			MATHARZ_INLINE template_quaternion<number_t> friend operator * (const template_quaternion<number_t> q, const template_quaternion<number_t> p) {
				return q.Multiply(p);
			}

			MATHARZ_INLINE void friend operator *= (const template_quaternion<number_t>& q, const template_quaternion<number_t> p) {
				q = q.Multiply(p);
			}

			MATHARZ_INLINE template_vec3<number_t> friend operator*(const template_quaternion<number_t> q, const template_vec3<number_t> v) {
				q.Multiply(v);
			};

			MATHARZ_INLINE template_vec3<number_t> friend operator*(const template_vec3<number_t> v, const template_quaternion<number_t> q) {
				q.Multiply(v);
			};

			// Get Inverse quaternion
			// @return inversed Quaternion
			MATHARZ_INLINE template_quaternion< number_t> GetInverse() const {
				return GetConjugate().Normalize();
			};

			// Get Inverse quaternion
			// @return inversed Quaternion
			MATHARZ_INLINE template_quaternion< number_t> Inverse() {
				return Conjugate().Normalize();
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_quaternion< number_t> quat)
			{
				return equal(x, quat.x) && equal(y, quat.y) && equal(z, quat.z) && equal(w, quat.w);
			};

			MATHARZ_INLINE bool operator==(template_vec4< number_t> vec4)
			{
				return equal(x, vec4.x) && equal(y, vec4.y) && equal(z, vec4.z) && equal(w, vec4.w);
			};

			// Access operators
			MATHARZ_INLINE number_t& operator[](size_t i) { MATHARZ_ASSERT((i < 4), "Out of size index"); return data[i]; };

			MATHARZ_INLINE const number_t& operator[](size_t i) const { MATHARZ_ASSERT((i < 4), "Out of size index"); return data[i]; };
		};

		// Matrix with 2x2 elemets
		template<typename number_t>
		struct template_matrix2x2 {
		public:
			// make alignas for avoiding bug(memory corruption?) in debug mode(??)
			union {
				struct {
					alignas(sizeof(number_t) * 4) number_t data[2][2];
				};
				struct {
					alignas(sizeof(number_t) * 4) number_t lineArrayData[4];
				};
				struct {
					template_vec2<number_t> vectors[2];
				};
				struct {
					alignas(sizeof(number_t) * 2) template_vec2<number_t> xVector;
					alignas(sizeof(number_t) * 2) template_vec2<number_t> yVector;
				};
				struct {
					alignas(sizeof(number_t) * 2) template_vec2<number_t> rVector;
					alignas(sizeof(number_t) * 2) template_vec2<number_t> gVector;
				};
				struct {
					number_t
						a, b,
						c, d;
				};
				struct {
					number_t
						a11, a12,
						a21, a22;
				};
			};

			template_matrix2x2<number_t>() { std::fill_n(&lineArrayData[0], 4, 0); };

			template_matrix2x2<number_t>(number_t aE, number_t bE,
				number_t cE, number_t dE)
			{
				a = aE; b = bE; c = cE; d = dE;
			};

			template_matrix2x2<number_t>(number_t Pivot)
			{
				std::fill_n(&lineArrayData[0], 4, 0);
				data[0][0] = Pivot;
				data[1][1] = Pivot;
			};

			template_matrix2x2<number_t>(template_vec2<number_t> aVec, template_vec2<number_t> bVec)
			{
				xVector = aVec;
				yVector = bVec;
			}

			template_matrix2x2<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 4, "Init list doesn't a proper size(>4 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 4, 0);
					data[0][0] = *initList.begin();
					data[1][1] = *initList.begin();
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 4), lineArrayData);
				}
			}

			explicit template_matrix2x2<number_t>(template_vec2<number_t> vec2s[2])
			{
				std::copy_n(&vec2s[0].data[0], 2, &data[0][0]);
				std::copy_n(&vec2s[1].data[0], 2, &data[1][0]);
			}

			explicit template_matrix2x2<number_t>(number_t dataE[4])
			{
				std::copy_n(&dataE[0], 4, &lineArrayData[0]);
			};

			explicit template_matrix2x2<number_t>(number_t dataE[2][2])
			{
				std::copy_n(&dataE[0][0], 4, &data[0][0]);
			};

			// Cast to another matrix2x2 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_matrix2x2< different_number_t>() const
			{
				template_matrix2x2<different_number_t> result;
				std::transform(this->lineArrayData, this->lineArrayData + 4, result.lineArrayData, [](number_t value) {return static_cast<different_number_t>(value); });
				return result;
			};

			// Multiply this matrix by another matrix b
			// @return Result matrix
			MATHARZ_INLINE template_matrix2x2<number_t> Multiply(template_matrix2x2<number_t> b) const
			{
				template_matrix2x2<number_t> result{};

				result[0][0] = data[0][0] * b.data[0][0] + data[0][1] * b.data[1][0];
				result[0][1] = data[0][0] * b.data[0][1] + data[0][1] * b.data[1][1];
				result[1][0] = data[1][0] * b.data[0][0] + data[1][1] * b.data[1][0];
				result[1][1] = data[1][0] * b.data[0][1] + data[1][1] * b.data[1][1];
				return result;
			}

			// Divide this matrix by scalar
			// @return Result matrix
			MATHARZ_INLINE template_matrix2x2<number_t> ScalarDivide(number_t b) const
			{
				MATHARZ_ASSERT(b != number_t(0), "Divide on 0!");
				template_matrix2x2<number_t> result{};
				if (b != number_t(0))
				{
					result[0][0] /= b;
					result[0][1] /= b;
					result[1][0] /= b;
					result[1][1] /= b;
				};
				return result;
			}

			// Divide this matrix by scalar
			// @return Result matrix
			MATHARZ_INLINE template_matrix2x2<number_t>& ScalarDivideSelf(number_t b)
			{
				MATHARZ_ASSERT(b != number_t(0), "Divide on 0!");
				if (b != number_t(0))
				{
					data[0][0] /= b;
					data[0][1] /= b;
					data[1][0] /= b;
					data[1][1] /= b;
				};
				return *this;
			}

			// Calculate determinant of this matrix
			// @return Determinant
			MATHARZ_INLINE number_t Determinant() const
			{
				return a11 * a22 - a12 * a21;
			}

			// Calculate inverse of this matrix
			// @return Inverse of this matrix, be sure to call this on invertible matrix, else use GetInverseSafe
			MATHARZ_INLINE template_matrix2x2<number_t> GetInverse() const
			{
				template_matrix2x2<number_t> result
				{ d, -b,
				 -c, a };
				return result.ScalarDivideSelf(result.Determinant());
			}

			// Calculate inverse of this matrix
			// @return Inverse of this matrix, if not invertible returns zero matrix
			MATHARZ_INLINE template_matrix2x2<number_t> GetInverseSafe(const number_t safe_gap = number_t(0.499999999999)) const
			{
				MATHARZ_ASSERT(abs(Determinant()) > number_t(safe_gap), "Matrix is not Invetrible!");

				if (abs<number_t>(Determinant()) > number_t(safe_gap))
					return GetInverse();

				return template_matrix2x2<number_t>{0};
			}

			// Inverse this matrix
			MATHARZ_INLINE template_matrix2x2<number_t>& InverseSelf()
			{
				ScalarDivideSelf(Determinant());
				return *this;
			}

			// Inverse this matrix with safe gap
			MATHARZ_INLINE template_matrix2x2<number_t>& InverseSelfSafe(const number_t safe_gap = number_t(0.499999999999))
			{
				MATHARZ_ASSERT(abs(Determinant()) > number_t(safe_gap), "Matrix is not Invetrible!");

				if (abs(Determinant()) > number_t(safe_gap))
				{
					ScalarDivideSelf(Determinant());
				}
				return *this;
			}

			// Math operators
			// Matrix multiply
			MATHARZ_INLINE template_matrix2x2<number_t> operator*(template_matrix2x2<number_t> b) const
			{
				return Multiply(b);
			}

			// Scalar divide
			MATHARZ_INLINE template_matrix2x2<number_t>&& operator/(template_matrix2x2<number_t> b) const
			{
				return ScalarDivide(b);
			}

			// Scalar divide self
			MATHARZ_INLINE template_matrix2x2<number_t>& operator/=(template_matrix2x2<number_t> b)
			{
				return ScalarDivideSelf(b);
			}

			MATHARZ_INLINE template_matrix2x2<number_t> operator +(const template_matrix2x2<number_t> b)
			{
				template_matrix2x2<number_t> res;
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						res.data[i][j] = data[i][j] + b.data[i][j];
					}
				}
				return res;
			}

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix2x2< number_t>& mat2)
			{
				return equal(lineArrayData[0], mat2.lineArrayData[0]) && equal(lineArrayData[1], mat2.lineArrayData[1])
					&& equal(lineArrayData[2], mat2.lineArrayData[2]) && equal(lineArrayData[3], mat2.lineArrayData[3]);
				//return this->vectors[0] == mat2.vectors[0] && this->vectors[1] == mat2.vectors[1];
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix3x3< number_t> mat3)
			{
				return vectors[0].x == mat3.vectors[0].x && vectors[0].y == mat3.vectors[0].y &&
					vectors[1].x == mat3.vectors[1].x && vectors[1].y == mat3.vectors[1].y;
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix4x4< number_t> mat4)
			{
				return vectors[0].x == mat4.vectors[0].x && vectors[0].y == mat4.vectors[0].y &&
					vectors[1].x == mat4.vectors[1].x && vectors[1].y == mat4.vectors[1].y;
			};

			// Access operators
			MATHARZ_INLINE template_vec2<number_t>& operator[](size_t i) { MATHARZ_CHECK(i < 2); return vectors[i]; };
			MATHARZ_INLINE const template_vec2<number_t>& operator[](size_t i) const { MATHARZ_CHECK(i < 2); return vectors[i]; };
		};

		// Matrix with 3x3 elemets
		template<typename number_t>
		struct template_matrix3x3 {
		public:
			union {
				number_t data[3][3];
				number_t rowData[3][3];
				number_t lineArrayData[9];
				template_vec3<number_t> vectors[3];
				struct {
					template_vec3<number_t> xVector;
					template_vec3<number_t> yVector;
					template_vec3<number_t> zVector;
				};
				struct {
					template_vec3<number_t> rVector;
					template_vec3<number_t> gVector;
					template_vec3<number_t> bVector;
				};
				struct {
					number_t a, b, c, d, e, f, g, h, i;
				};
				struct {
					number_t a11, a12, a13, a21, a22, a23, a31, a32, a33;
				};
			};

			template_matrix3x3<number_t>() { std::fill_n(&lineArrayData[0], 9, 0); };

			template_matrix3x3<number_t>(number_t aE, number_t bE, number_t cE,
				number_t dE, number_t eE, number_t fE,
				number_t gE, number_t hE, number_t iE)
			{
				a = aE; b = bE; c = cE; d = dE; e = eE; f = fE; g = gE; h = hE; i = iE;;
			};

			template_matrix3x3<number_t>(number_t Pivot)
			{
				std::fill_n(&lineArrayData[0], 9, 0);
				data[0][0] = Pivot;
				data[1][1] = Pivot;
				data[2][2] = Pivot;
			};

			template_matrix3x3<number_t>(template_vec3<number_t> vec1, template_vec3<number_t> vec2, template_vec3<number_t> vec3)
			{
				vectors[0] = vec1;
				vectors[1] = vec2;
				vectors[2] = vec3;
			};

			template_matrix3x3<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 9, "Init list doesn't a proper size(>9 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 9, 0);
					data[0][0] = *initList.begin();
					data[1][1] = *initList.begin();
					data[2][2] = *initList.begin();
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 9), lineArrayData);
				}
			}

			explicit template_matrix3x3<number_t>(template_vec3<number_t> vec3s[3])
			{
				std::copy_n(&vec3s.data[0], 3, &data[0][0]);
				std::copy_n(&vec3s.data[1], 3, &data[1][0]);
				std::copy_n(&vec3s.data[2], 3, &data[2][0]);
			}

			explicit template_matrix3x3<number_t>(number_t dataE[9])
			{
				std::copy_n(&dataE[0], 9, &lineArrayData[0]);
			};

			explicit template_matrix3x3<number_t>(number_t dataE[3][3])
			{
				std::copy_n(&dataE[0][0], 9, &data[0][0]);
			};

			// Cast to another matrix3x3 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_matrix3x3< different_number_t>() const
			{
				template_matrix3x3<different_number_t> result;
				std::transform(this->lineArrayData, this->lineArrayData + 9, result.lineArrayData, [](number_t value) {return static_cast<different_number_t>(value); });
				return result;
			};

			// Multiply this matrix by another matrix b
			MATHARZ_INLINE template_matrix3x3<number_t> Multiply(template_matrix3x3<number_t> b)
			{
				template_matrix3x3<number_t> result{};
				for (int i = 0; i < 3; ++i)
					for (int j = 0; j < 3; ++j)
						for (int k = 0; k < 3; ++k)
						{
							result.data[i][j] += this->data[i][k] * b.data[k][j];
						};

				return result;
			}

			// Multiply this matrix by another matrix b
			MATHARZ_INLINE template_matrix3x3<number_t> Multiply(number_t b)
			{
				template_matrix3x3<number_t> result{};
				for (int i = 0; i < 3; ++i)
					for (int j = 0; j < 3; ++j)
							result.data[i][j] = this->data[i][j] * b;
						
				return result;
			}

			// Calculate determinant of this matrix
			MATHARZ_INLINE number_t Determinant()
			{
				number_t determinant = 0;

				determinant += a11 * ((a22 * a33) - (a23 * a32));
				determinant -= a12 * ((a21 * a33) - (a23 * a31));
				determinant += a13 * ((a21 * a32) - (a22 * a31));

				return determinant;
			}

			// Addition
			// @return Vector with difference of each elements of vectors
			MATHARZ_INLINE template_matrix3x3<number_t> operator+(const template_matrix3x3<number_t> b) const
			{
				return template_matrix3x3<number_t> {
					a11 + b.a11, a12 + b.a12, a13 + b.a13,
					a21 + b.a21, a22 + b.a22, a23 + b.a23,
					a31 + b.a31, a32 + b.a32, a33 + b.a33};
			}

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix2x2< number_t> mat2)
			{
				return vectors[0].xy == mat2.vectors[0] && vectors[1].xy == mat2.vectors[1];
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix3x3< number_t> mat3)
			{
				return vectors[0] == mat3.vectors[0] && vectors[1] == mat3.vectors[1]
					&& vectors[2] == mat3.vectors[2];
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix4x4< number_t> mat4)
			{
				return vectors[0] == mat4.vectors[0].xyz && vectors[1] == mat4.vectors[1].xyz
					&& vectors[2] == mat4.vectors[2].xyz;
			};

			// Access operators
			MATHARZ_INLINE template_vec3<number_t>& operator[](size_t i) { MATHARZ_CHECK(i < 3); return vectors[i]; };
			MATHARZ_INLINE const template_vec3<number_t>& operator[](size_t i) const { MATHARZ_CHECK(i < 3); return vectors[i]; };
		};

		// Matrix with 4x4 elemets
		template<typename number_t>
		struct template_matrix4x4 {
		public:
			union {
				number_t data[4][4];
				number_t rowData[4][4];
				number_t lineArrayData[16];
				template_vec4<number_t> vectors[4];
				struct {
					template_vec4<number_t> xVector;
					template_vec4<number_t> yVector;
					template_vec4<number_t> zVector;
					template_vec4<number_t> wVector;
				};
				struct {
					template_vec4<number_t> rVector;
					template_vec4<number_t> gVector;
					template_vec4<number_t> bVector;
					template_vec4<number_t> aVector;
				};
				struct {
					number_t
						a, b, c, d,
						e, f, g, h,
						i, j, k, l,
						m, n, o, p;
				};
				struct {
					number_t
						a11, a12, a13, a14,
						a21, a22, a23, a24,
						a31, a32, a33, a34,
						a41, a42, a43, a44;
				};
			};

			template_matrix4x4<number_t>(
				number_t aE, number_t bE, number_t cE, number_t dE = 0,
				number_t eE = 0, number_t fE = 0, number_t gE = 0, number_t hE = 0,
				number_t iE = 0, number_t jE = 0, number_t kE = 0, number_t lE = 0,
				number_t mE = 0, number_t nE = 0, number_t oE = 0, number_t pE = 0)
			{
				a = aE; b = bE; c = cE; d = dE;
				e = eE; f = fE; g = gE; h = hE;
				i = iE; j = jE; k = kE; l = lE;
				m = mE; n = nE; o = oE; p = pE;
			};

			template_matrix4x4<number_t>() { std::fill_n(&lineArrayData[0], 16, 0); };

			template_matrix4x4<number_t>(number_t Pivot)
			{
				std::fill_n(&lineArrayData[0], 16, 0);
				data[0][0] = Pivot;
				data[1][1] = Pivot;
				data[2][2] = Pivot;
				data[3][3] = Pivot;
			};

			template_matrix4x4<number_t>(template_vec4<number_t> vec1, template_vec4<number_t> vec2, template_vec4<number_t> vec3, template_vec4<number_t> vec4)
			{
				vectors[0] = vec1;
				vectors[1] = vec2;
				vectors[2] = vec3;
				vectors[3] = vec4;
			};

			template_matrix4x4<number_t>(std::initializer_list<number_t> initList)
			{
				MATHARZ_ASSERT(initList.size() <= 16, "Init list doesn't a proper size(>16 numbers)!");

				if (initList.size() == 1)
				{
					std::fill_n(&lineArrayData[0], 16, 0);
					data[0][0] = *initList.begin();
					data[1][1] = *initList.begin();
					data[2][2] = *initList.begin();
					data[3][3] = *initList.begin();
				}
				else
				{
					std::copy_n(initList.begin(), matharz::clamp((int)(initList.end() - initList.begin()), 0, 16), &lineArrayData[0]);
				};
			};

			explicit template_matrix4x4<number_t>(template_vec4<number_t> vec4s[4])
			{
				for (int i = 0; i < 3; i++)
				{
					vectors[i] = vec4s[i];
				}
			};
			explicit template_matrix4x4<number_t>(template_matrix3x3<number_t> mat3x3, template_vec3<number_t> wVec3Right, template_vec4<number_t> vec4Down)
			{
				//	max3x3[0][0,	1,	2],wVec3Right[	0]
				//	max3x3[1][0,	1,	2],wVec3Right[	1]
				//	max3x3[2][0,	1,	2],wVec3Right[	2]
				//	wVec4Down[0,	1,	2,				3]

				std::copy_n(&mat3x3.data[0][0], 3, &data[0][0]);
				data[0][3] = wVec3Right.data[0];

				std::copy_n(&mat3x3.data[1][0], 3, &data[1][0]);
				data[1][3] = wVec3Right.data[1];

				std::copy_n(&mat3x3.data[2][0], 3, &data[2][0]);
				data[2][3] = wVec3Right.data[2];

				std::copy_n(&vec4Down.data[0], 4, &data[3][0]);
			};

			explicit template_matrix4x4<number_t>(number_t dataE[16])
			{
				std::copy_n(&dataE[0], 16, &lineArrayData[0]);
			};
			explicit template_matrix4x4<number_t>(number_t dataE[4][4])
			{
				std::copy_n(&dataE[0][0], 16, &data[0][0]);
			}

			// Cast to another matrix3x3 template, be careful about with data you cast, possible lose of data
			template <typename different_number_t>
			explicit operator template_matrix4x4< different_number_t>() const
			{
				template_matrix4x4<different_number_t> result{};
				std::transform(this->lineArrayData, this->lineArrayData + 16, result->lineArrayData, [](number_t value) {return static_cast<different_number_t>(value); });
				return result;
			};

			// Calculate determinant of this matrix
			// @return Determinant number
			MATHARZ_INLINE number_t Determinant()
			{
				// Solution from ref https://stackoverflow.com/questions/1276611/which-method-of-matrix-determinant-calculation-is-this

				// 2x2 sub-determinants
				number_t det2_01_01 = data[0][0] * data[1][1] - data[0][1] * data[1][0];
				number_t det2_01_02 = data[0][0] * data[1][2] - data[0][2] * data[1][0];
				number_t det2_01_03 = data[0][0] * data[1][3] - data[0][3] * data[1][0];
				number_t det2_01_12 = data[0][1] * data[1][2] - data[0][2] * data[1][1];
				number_t det2_01_13 = data[0][1] * data[1][3] - data[0][3] * data[1][1];
				number_t det2_01_23 = data[0][2] * data[1][3] - data[0][3] * data[1][2];

				// 3x3 sub-determinants
				number_t det3_201_012 = data[2][0] * det2_01_12 - data[2][1] * det2_01_02 + data[2][2] * det2_01_01;
				number_t det3_201_013 = data[2][0] * det2_01_13 - data[2][1] * det2_01_03 + data[2][3] * det2_01_01;
				number_t det3_201_023 = data[2][0] * det2_01_23 - data[2][2] * det2_01_03 + data[2][3] * det2_01_02;
				number_t det3_201_123 = data[2][1] * det2_01_23 - data[2][2] * det2_01_13 + data[2][3] * det2_01_12;

				return (-det3_201_123 * data[3][0] + det3_201_023 * data[3][1] - det3_201_013 * data[3][2] + det3_201_012 * data[3][3]);
			}

			// Rotate contained 3d "information" of some object/thing with angle around axis from Normalized v vector
			// @return Matrix with corresponding information about rotation
			MATHARZ_INLINE template_matrix4x4<number_t> Rotate(number_t angle, const template_vec3<number_t>& v)
			{
				const number_t a = angle;
				const number_t c = matharz::cos(a);
				const number_t s = matharz::sin(a);

				template_vec3<number_t> axis(v.GetNormalized());
				template_vec3<number_t> temp(axis.ScalarMultiply(number_t(1) - c));

				number_t f11 = c + temp.data[0] * axis.data[0];
				number_t f12 = temp.data[0] * axis.data[1] + s * axis.data[2];
				number_t f13 = temp.data[0] * axis.data[2] - s * axis.data[1];

				number_t f21 = temp.data[1] * axis.data[0] - s * axis.data[2];
				number_t f22 = c + temp.data[1] * axis.data[1];
				number_t f23 = temp.data[1] * axis.data[2] + s * axis.data[0];

				number_t f31 = temp.data[2] * axis.data[0] + s * axis.data[1];
				number_t f32 = temp.data[2] * axis.data[1] - s * axis.data[0];
				number_t f33 = c + temp.data[2] * axis.data[2];


				auto ResultVec1 = vectors[0].ScalarMultiply(f11) + vectors[1].ScalarMultiply(f12) + vectors[2].ScalarMultiply(f13);
				auto ResultVec2 = vectors[0] * f21 + vectors[1] * f22 + vectors[2] * f23;
				auto ResultVec3 = vectors[0] * f31 + vectors[1] * f32 + vectors[2] * f33;
				auto ResultVec4 = vectors[3];

				return template_matrix4x4<number_t>{ResultVec1, ResultVec2, ResultVec3, ResultVec4};
			};

			MATHARZ_INLINE template_vec4<number_t> Multiply(const template_vec4<number_t> b) const
			{
				// use overloaded matrix opeartor* for vec4
				return *this * b;
			}
			// Operators

			// Multiplication
			// @return Matrix with results of multiplication of each columns with corresponding elements of vector
			MATHARZ_INLINE template_vec4<number_t> operator*(const template_vec4<number_t> b) const
			{
				template_vec4<number_t> Add0;
				template_vec4<number_t> Add1;
				{
					const template_vec4<number_t> Mov0(b.data[0]);
					const template_vec4<number_t> Mov1(b.data[1]);
					const template_vec4<number_t> Mul0 = this->vectors[0] * Mov0;
					const template_vec4<number_t> Mul1 = this->vectors[1] * Mov1;
					Add0 = Mul0 + Mul1;
				}
				{
					const template_vec4<number_t> Mov2(b.data[2]);
					const template_vec4<number_t> Mov3(b.data[3]);
					const template_vec4<number_t> Mul2 = this->vectors[2] * Mov2;
					const template_vec4<number_t> Mul3 = this->vectors[3] * Mov3;
					Add1 = Mul2 + Mul3;
				}
				return std::move(Add0 + Add1);
			}

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix2x2< number_t> mat2)
			{
				return vectors[0].xy == mat2.vectors[0] && vectors[1].xy == mat2.vectors[1];
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix3x3< number_t> mat3)
			{
				return vectors[0].xyz == mat3.vectors[0] && vectors[1].xyz == mat3.vectors[1]
					&& vectors[2].xyz == mat3.vectors[2];
			};

			// bool operators
			MATHARZ_INLINE bool operator==(template_matrix4x4< number_t> mat4)
			{
				return vectors[0] == mat4.vectors[0] && vectors[1] == mat4.vectors[1]
					&& vectors[2] == mat4.vectors[2] && vectors[3] == mat4.vectors[3];
			};

			// Access operators
			MATHARZ_INLINE template_vec4<number_t>& operator[](size_t i) { MATHARZ_CHECK(i < 4); return this->vectors[i]; };

			MATHARZ_INLINE const template_vec4<number_t>& operator[](size_t i) const { MATHARZ_CHECK(i < 4); return this->vectors[i]; };
		};

		// std << operators overload for math classes
		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_vec2<number_t> vec)
		{
			os << '[' << vec.x << ':' << vec.y << ']';
			return os;
		};

		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_vec3<number_t> vec)
		{
			os << '[' << vec.x << ':' << vec.y << ':' << vec.z << ']';
			return os;
		};

		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_vec4<number_t> vec)
		{
			os << '[' << vec.x << ':' << vec.y << ':' << vec.z << ':' << vec.w << ']';
			return os;
		};

		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_quaternion<number_t> quat)
		{
			os << quat.vector;
			return os;
		};

		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_matrix2x2<number_t> mat2x2)
		{
			os << '[' << mat2x2.data[0][0] << ':' << mat2x2.data[0][1] << ']' << '\n' <<
				'[' << mat2x2.data[1][0] << ':' << mat2x2.data[1][1] << ']' << '\n';

			return os;
		};

		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_matrix3x3<number_t> mat3x3)
		{
			os << '[' << mat3x3.data[0][0] << ':' << mat3x3.data[0][1] << ':' << mat3x3.data[0][2] << ']' << '\n' <<
				'[' << mat3x3.data[1][0] << ':' << mat3x3.data[1][1] << ':' << mat3x3.data[1][2] << ']' << '\n' <<
				'[' << mat3x3.data[2][0] << ':' << mat3x3.data[2][1] << ':' << mat3x3.data[2][2] << ']' << '\n';
			return os;
		};


		template <typename number_t>
		std::ostream& operator<<(std::ostream& os, const matharz::template_matrix4x4<number_t> mat4x4)
		{
			os << '[' << mat4x4.data[0][0] << ':' << mat4x4.data[0][1] << ':' << mat4x4.data[0][2] << ':' << mat4x4.data[0][3] << ']' << '\n' <<
				'[' << mat4x4.data[1][0] << ':' << mat4x4.data[1][1] << ':' << mat4x4.data[1][2] << ':' << mat4x4.data[1][3] << ']' << '\n' <<
				'[' << mat4x4.data[2][0] << ':' << mat4x4.data[2][1] << ':' << mat4x4.data[2][2] << ':' << mat4x4.data[2][3] << ']' << '\n' <<
				'[' << mat4x4.data[3][0] << ':' << mat4x4.data[3][1] << ':' << mat4x4.data[3][2] << ':' << mat4x4.data[3][3] << ']' << '\n';
			return os;
		};

		// left-side operators overloads
		template <typename number_t>
		MATHARZ_INLINE template_vec2<number_t> operator/ (number_t a, template_vec2<number_t> vec2)
		{
			return { a / vec2.x, a / vec2.y };
		}

		template <typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator/ (number_t a, template_vec3<number_t> vec3)
		{
			return { a / vec3.x, a / vec3.y, a / vec3.z };
		}

		template <typename number_t>
		MATHARZ_INLINE template_vec4<number_t> operator/ (number_t a, template_vec4<number_t> vec4)
		{
			return { a / vec4.x, a / vec4.y, a / vec4.z, a / vec4.w };
		}

		template <typename number_t>
		MATHARZ_INLINE template_matrix2x2<number_t> operator/ (number_t a, template_matrix2x2<number_t> mat2)
		{
			return { a / mat2.xVector, a / mat2.yVector };
		}

		template <typename number_t>
		MATHARZ_INLINE template_matrix3x3<number_t> operator/ (number_t a, template_matrix3x3<number_t> nat3)
		{
			return { a / nat3.xVector, a / nat3.yVector, a / nat3.zVector };
		}

		template <typename number_t>
		MATHARZ_INLINE template_matrix4x4<number_t> operator/ (number_t a, template_matrix4x4<number_t> mat4)
		{
			return { a / mat4.xWector, a / mat4.yWector, a / mat4.zWector, a / mat4.wWector };
		}

		template <typename number_t>
		MATHARZ_INLINE template_vec2<number_t> operator+ (number_t a, template_vec2<number_t> vec2)
		{
			return { a + vec2.x, a + vec2.y };
		}

		template <typename number_t>
		MATHARZ_INLINE template_vec3<number_t> operator+ (number_t a, template_vec3<number_t> vec3)
		{
			return { a + vec3.x, a + vec3.y, a + vec3.z };
		}

		template <typename number_t>
		MATHARZ_INLINE template_vec4<number_t> operator+ (number_t a, template_vec4<number_t> vec4)
		{
			return { a + vec4.x, a + vec4.y, a + vec4.z, a + vec4.w };
		}

		template<typename number_t> MATHARZ_INLINE bool IsPerpendicular(template_vec3<number_t> vecA, template_vec3<number_t> vecB) {
			return (vecA.IsPerpendicularTo(vecB));
		};

		template<typename number_t> MATHARZ_INLINE bool IsParallel(template_vec3<number_t> vecA, template_vec3<number_t> vecB) {
			return (vecA.IsParallelTo(vecB));
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Add(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft + rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> ScalarAdd(template_vec3<number_t> lft, number_t rgt) {
			return (lft + rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Subtract(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft - rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> ScalarSubtract(template_vec3<number_t> lft, number_t rgt) {
			return (lft - rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Multiply(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft * rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> ScalarMultiply(template_vec3<number_t> lft, number_t rgt) {
			return (lft * rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Divide(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft / rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> ScalarDivide(template_vec3<number_t> lft, number_t rgt) {
			return (lft / rgt);
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> ProjectionFrom(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft.ProjectionFrom(rgt));
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Projection(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft.Projection(rgt));
		};

		template<typename number_t> MATHARZ_INLINE number_t DotProduct(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft.DotProduct(rgt));
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> CrossProduct(template_vec3<number_t> lft, template_vec3<number_t> rgt) {
			return (lft.CrossProduct(rgt));
		};

		template<typename number_t> MATHARZ_INLINE number_t SquareLength(template_vec3<number_t> vec) {
			return vec.SquareLength();
		};

		template<typename number_t> MATHARZ_INLINE number_t SquareMagnitude(template_vec3<number_t> vec) {
			return vec.SquareMagnitude();
		};

		template<typename number_t> MATHARZ_INLINE number_t Length(template_vec3<number_t> vec) {
			return vec.Length();
		};

		template<typename number_t> MATHARZ_INLINE number_t Magnitude(template_vec3<number_t> vec) {
			return vec.Magnitude();
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Inverse(template_vec3<number_t> vec) {
			return vec.GetInverse();
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Normalize(template_vec3<number_t> vec) {
			return vec.GetNormalized();
		};

		template<typename number_t> MATHARZ_INLINE template_vec3<number_t> Lerp(template_vec3<number_t> a, template_vec3<number_t> b, number_t t) {
			MATHARZ_CHECK(t >= 0.f && t <= 1.f);
			return (1.f - t) * a + t * b;
		};

		template<typename number_t, typename angle_t = float>
		MATHARZ_INLINE template_quaternion<number_t>
			MakeQuaternionFromAxisAngle(template_vec3<number_t>  axis, angle_t angle)
		{
			template_quaternion<number_t> result{};
			result.MakeFromAxisAngle(axis, angle);
			return result;
		};

		template<typename number_t>
		MATHARZ_INLINE template_quaternion<number_t>
			MakeQuaternionFromEulerRotation(template_vec3<number_t> vec)
		{
			template_quaternion<number_t> result{};
			result.MakeFromEulerRotation(vec);
			return result;
		};

		template<typename number_t, typename float_t = float>
		MATHARZ_INLINE template_quaternion<number_t>
			MakeQuaternionFromEulerRotation(float_t x, float_t y, float_t z)
		{
			template_quaternion<number_t> result{};
			result.MakeFromEulerRotation(x, y, z);
			return result;
		};

		template<typename number_t> MATHARZ_INLINE template_matrix4x4<number_t> Rotate(template_matrix4x4<number_t> m, number_t angle, template_vec3<number_t> vec) {
			return m.Rotate(angle, vec);
		};

		// Static constants
		MATHARZ_STATIC_GLOBAL mat2x2 IdentityMatrix2x2{
			1, 0,
			0, 1
		};

		MATHARZ_STATIC_GLOBAL mat3x3 IdentityMatrix3x3{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		};

		MATHARZ_STATIC_GLOBAL mat4x4 IdentityMatrix4x4{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		};

		MATHARZ_STATIC_GLOBAL quat IndentityQuaternion{
			0.f,0.f,0.f,1.f
		};

	} // math
} // harz

// Hash std implementation for math classes
namespace std {

	template <typename number_t> struct hash<matharz::template_vec2<number_t>>
	{
		size_t operator()(const matharz::template_vec2<number_t>& vec) const
		{
			return hash<size_t>()(hash<number_t>()(vec.x) + hash<number_t>()(vec.y));
		}
	};

	template <typename number_t> struct hash<matharz::template_vec3<number_t>>
	{
		size_t operator()(const matharz::template_vec3<number_t>& vec) const
		{
			return hash<size_t>()(hash<number_t>()(vec.x) + hash<number_t>()(vec.y) + hash<number_t>()(vec.z));
		}
	};

	template <typename number_t> struct hash<matharz::template_vec4<number_t>>
	{
		size_t operator()(const matharz::template_vec4<number_t>& vec) const
		{
			return hash<size_t>()(hash<number_t>()(vec.x) + hash<number_t>()(vec.y) + hash<number_t>()(vec.z) + hash<number_t>()(vec.w));
		}
	};

	template <typename number_t> struct hash<matharz::template_quaternion<number_t>>
	{
		size_t operator()(const matharz::template_quaternion<number_t>& quat) const
		{
			return hash<matharz::template_vec4<number_t>>()(quat.vector);
		}
	};


	template <typename number_t> struct hash<matharz::template_matrix2x2<number_t>>
	{
		size_t operator()(const matharz::template_matrix2x2<number_t>& mat2) const
		{
			return hash<size_t>()(hash<matharz::template_vec2<number_t>>()(mat2.xVector) + hash<matharz::template_vec2<number_t>>()(mat2.yVector));
		}
	};

	template <typename number_t> struct hash<matharz::template_matrix3x3<number_t>>
	{
		size_t operator()(const matharz::template_matrix3x3<number_t>& mat3) const
		{
			return hash<size_t>()(hash<matharz::template_vec3<number_t>>()(mat3.xVector) + hash<matharz::template_vec3<number_t>>()(mat3.yVector) + hash<matharz::template_vec3<number_t>>()(mat3.zVector));
		}
	};

	template <typename number_t> struct hash<matharz::template_matrix4x4<number_t>>
	{
		size_t operator()(const matharz::template_matrix4x4<number_t>& mat4) const
		{
			return hash<size_t>()(hash<matharz::template_vec4<number_t>>()(mat4.xVector) + hash<matharz::template_vec4<number_t>>()(mat4.yVector) +
				hash<matharz::template_vec4<number_t>>()(mat4.zVector) + hash<matharz::template_vec4<number_t>>()(mat4.wVector));
		}
	};
}
#endif // !MATHARZ HEADER
