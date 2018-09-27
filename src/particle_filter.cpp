#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "config.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double gps_std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// Set number of particles
	int num_particles = ConfigParams::numOfParticles;
	particles.resize(num_particles);
	weights.resize(num_particles, 1.0);
	
	// Gaussian 
	default_random_engine gen;
	normal_distribution<double> dist_x(gps_x, gps_std[0]);
	normal_distribution<double> dist_y(gps_y, gps_std[1]);
	normal_distribution<double> dist_theta(gps_theta, gps_std[2]);

	for(int i = 0; i < num_particles; ++i)
	{
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
	}

	// Set the initialization flag to true;
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	
	default_random_engine gen;
	// Zero mean
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);


	// Update the position
	for(int i = 0; i < num_particles; ++i)
	{
		auto& x = particles[i].x;
		auto& y = particles[i].y;
		auto& theta = particles[i].theta;

		// Check the yaw_rate
		if(yaw_rate < 0.0001)
		{
			x += velocity * delta_t * cos(theta) + dist_x(gen);
			y += velocity * delta_t * sin(theta) + dist_y(gen);
			theta += dist_theta(gen);
		}
		else
		{
			x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta)) + dist_x(gen);
			y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t)) + dist_y(gen);
			theta += yaw_rate * delta_t + dist_theta(gen);
		}

		// Normalize the angle
		theta = std::fmod(theta, 2.0 * M_PI);
	}
}

vector<AssocResult> ParticleFilter::dataAssociation(std::vector<LandmarkObs> tracked_observations, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	vector<AssocResult> result(observations.size());

	for(int i = 0; i < observations.size(); ++i)
	{
		const auto& x = observations[i].x;
		const auto& y = observations[i].y;
		const auto& id_obe = observations[i].id;
		double thres = numeric_limits<double>::max();
		for(int j = 0; j < tracked_observations.size(); ++j)
		{
			const auto& dx = tracked_observations[i].x - x;
			const auto& dy = tracked_observations[i].y - y;
			const auto& id_track = tracked_observations[i].id;
			double dist_square = dx * dx + dy * dy;


			if(dist_square < thres)
			{
				result[i].from = id_obe; // from observation
				result[i].to   = id_track; // to tracked observation
				thres = dist_square;// update threshod
			}
		}
	}
	return result;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	vector<LandmarkObs> observationsInLcs(observations.size()); // observation in local coordinate system
	vector<LandmarkObs> tracked_observations(map_landmarks.landmark_list.size());

	for(int i = 0; i < num_particles; ++i)
	{

		vector<AssocResult> result(observations.size());

		const auto& particle = particles[i];
		vcsToLcs(particle ,observations, observationsInLcs); // map into same coordinate system 
		mapToLandmark(map_landmarks, tracked_observations);
		result = dataAssociation(tracked_observations, observationsInLcs);

		// Update weights for each particle
		particles[i].weight = multiGaussianProbDensity(result, observationsInLcs, tracked_observations, std_landmark);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	
	// Sum up all the weights for calcaulate the percentatge of each particles
	double total_weight = 0.0;
	
	for(int i = 0; i < num_particles; ++i)
	{
		total_weight += particles[i].weight;
	}

	// Fill in the percentage of weight for each particle
	for(int i = 0; i < num_particles; ++i)
	{
		weights[i] = particles[i].weight / total_weight;
	}

	// Build the discrete distribution
	discrete_distribution<int> dist(weights.begin(), weights.end());

	std::vector<Particle> newParticles(num_particles);
	
	// Resampling
	for(int i = 0; i < num_particles; ++i)
	{
		int id = dist(gen); // generate the random id base on the weigth of each particle
		newParticles[i] = particles[id];
	}

	// Update
	particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::vcsToLcs(const Particle& particle, const vector<LandmarkObs>& observationsInVcs, vector<LandmarkObs>& observationsInLcs)
{
	//1. Assign the basic property
	observationsInLcs = observationsInVcs;

	//2. Build up the transformation matrix
	// | x |     | cos -sin x_t |   | x |
	// | y |  =  | sin cos  y_t | * | y |
	// | 1 |     | 0    0     1 |   | 1 | 
	//  lcs                           vcs

	//3. Transform all the measurements from VCS to LCS
	const auto& theta = particle.theta;
	const auto& x_t = particle.x;
	const auto& y_t = particle.y;
	
	for(int i = 0; i < observationsInLcs.size(); ++i)
	{
		const auto& x_vcs = observationsInVcs[i].x;
		const auto& y_vcs = observationsInVcs[i].y;
		observationsInLcs[i].x = cos(theta) * x_vcs - sin(theta) * y_vcs + x_t;
		observationsInLcs[i].y = sin(theta) * x_vcs + cos(theta) * y_vcs + y_t;
	}
}

void ParticleFilter::mapToLandmark(const Map& map, vector<LandmarkObs>& landmarks)
{
	landmarks.resize(map.landmark_list.size());

	for(int i = 0; i < landmarks.size(); ++i)
	{
		auto& mark = map.landmark_list[i];
		landmarks[i].id = mark.id_i;
		landmarks[i].x = mark.x_f;
		landmarks[i].y = mark.y_f;
	}
}


double ParticleFilter::multiGaussianProbDensity(const std::vector<AssocResult>& result,  
												const std::vector<LandmarkObs>& observations, 
												const std::vector<LandmarkObs>& tracked_observations, 
												const double* std_landmark)
{
	double prob = 1.0;
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1]; 

	// All the measurements 
	for(int i = 0; i < observations.size(); ++i)
	{
		// extract the id from result.
		const auto& from = result[i].from;
		const auto& to = result[i].to;

		// Here suppose to search the corresponding id inside both observations and tracked_observations
		// However, it is alreay known that the id is just the index.
		const auto& x = observations[from].x;
		const auto& y = observations[from].y;
		const auto& mu_x = tracked_observations[to].x;
		const auto& mu_y = tracked_observations[to].y;


		double gauss_norm = (1/(2*M_PI*std_x*std_y));
		double exponent = (x-mu_x)*(x-mu_x)/(2*std_x*std_x) + (y-mu_y)*(y-mu_y)/(2*std_y*std_y);
		double gaussain = exp(-exponent) * gauss_norm;

		prob *= gaussain;
		
	}

	return prob;
}