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
#define verbose_particles false
#define verbose_association false
#define verbose_weights false
#define verbose false

using namespace std;

void ParticleFilter::init(double gps_x, double gps_y, double gps_theta, double gps_std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// Set number of particles
	num_particles = ConfigParams::numOfParticles;
	particles.resize(num_particles);
	weights.resize(num_particles, 1.0);
	
	if(verbose)
	{
		cout<<"[init] num_particles="<<num_particles<<endl;
		cout<<"[init] gps: x="<<gps_x<<", y="<<gps_y<<", theta="<<gps_theta<<endl;
	}

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
		auto& p = particles[i];

		// Check the yaw_rate
		if(fabs(yaw_rate) < 0.001)
		{
			p.x += velocity * delta_t * cos(p.theta) + dist_x(gen);
			p.y += velocity * delta_t * sin(p.theta) + dist_y(gen);
			p.theta += dist_theta(gen);
		}
		else
		{
			double a = p.theta + yaw_rate * delta_t;
			p.x += (velocity / yaw_rate) * (sin(a) - sin(p.theta)) + dist_x(gen);
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(a)) + dist_y(gen);
			p.theta += yaw_rate * delta_t + dist_theta(gen);
		}

		// Normalize the angle
		//p.theta = std::fmod(p.theta, 2.0 * M_PI);

		if(verbose_particles)
		{
			cout<<"particle ["<<i<<"], x="<<p.x<<", y="<<p.y<<", theta="<<p.theta<<endl;
		}
		
	}
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	for(int i = 0; i < observations.size(); ++i)
	{
		const auto& x = observations[i].x;
		const auto& y = observations[i].y;
		observations[i].id = -1; // assign an inalid value
		double thres = numeric_limits<double>::max();

		for(int j = 0; j < landmarks.size(); ++j)
		{
			const auto& dx = landmarks[j].x - x;
			const auto& dy = landmarks[j].y - y;
			auto id_landmark = landmarks[j].id;
			double dist_square = dx * dx + dy * dy;


			if(dist_square < thres)
			{
				observations[i].id = id_landmark;  // use the empty observations[i].id to store the corresponding landmark id
				thres = dist_square;// update threshod
			}
		}
	}

	if(verbose_association)
	{
		for(int i = 0; i < observations.size(); ++i)
		{
			cout<<"observations ["<<i<<"], to="<<observations[i].id<<endl;
		}
	}

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
	
	for(int i = 0; i < num_particles; ++i)
	{
		vector<LandmarkObs> observationsInLcs(observations.size()); // observation in local coordinate system
		vector<LandmarkObs> landmarks; // the size will be determined later
		const auto& particle = particles[i];
		
		// Transform observations from VCS to LCS
		vcsToLcs(particle ,observations, observationsInLcs); // data association is in LCS
		
		// Extract lane markers with sensor range
		mapToLandmark(particle, sensor_range, map_landmarks, landmarks); // use particle position to filter out based on sensor range
		
		// Association observations -> landmarks
		dataAssociation(landmarks, observationsInLcs);

		// Update weights for each particle
		if(verbose) {cout<<"[updateWeights] particle ["<<i<<"]:"<<endl;}
		particles[i].weight = multiGaussianProbDensity(observationsInLcs, landmarks, std_landmark);
		//particles[i].weight = mulGau(observationsInLcs, map_landmarks, std_landmark);

		if(verbose_weights)
		{
			cout<<"particle ["<<i<<"] weight="<<particles[i].weight<<endl;
		}
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
	particles.swap(newParticles);
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
	for(int i = 0; i < observationsInVcs.size(); ++i)
	{
		const auto& ob_vcs = observationsInVcs[i];
		observationsInLcs[i].x = cos(particle.theta) * ob_vcs.x - sin(particle.theta) * ob_vcs.y + particle.x;
		observationsInLcs[i].y = sin(particle.theta) * ob_vcs.x + cos(particle.theta) * ob_vcs.y + particle.y;
	}
}

void ParticleFilter::mapToLandmark(const Particle& p, const int& sensorRange, const Map& map, vector<LandmarkObs>& landmarks)
{

	for(int i = 0; i < map.landmark_list.size(); ++i)
	{
		auto mark = map.landmark_list[i];

		// Test the sensor range
		auto distance = dist(p.x, p.y, mark.x_f, mark.y_f);
		if(distance <= sensorRange) 
		{
			LandmarkObs obs;
			obs.id = mark.id_i;
			obs.x  = mark.x_f;
			obs.y  = mark.y_f;
			landmarks.push_back(obs);
		}
	}
}


double ParticleFilter::multiGaussianProbDensity(const vector<LandmarkObs>& observations, 
												const vector<LandmarkObs>& landmarks, 
												const double* std_landmark)
{
	double prob = 1.0;
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1]; 
	if(verbose)
	{
		cout<<"[multiGaussianProbDensity] observations.size()="<<observations.size()<<endl;
		cout<<"[multiGaussianProbDensity] landmarks.size()="<<landmarks.size()<<endl;
	}

	// All the measurements 
	for(int i = 0; i < observations.size(); ++i)
	{
		for(int j = 0; j < landmarks.size(); ++j)
		{
			// Here suppose to search the corresponding id inside both observations and tracked_observations
			if(observations[i].id == landmarks[j].id)
			{
				
				const auto x = observations[i].x;
				const auto y = observations[i].y;
				const auto mu_x = landmarks[j].x;
				const auto mu_y = landmarks[j].y;
				prob *=  multivariate_gaussian(x, y, mu_x, mu_y, std_x, std_y);
				continue;
			}
		}
	}
	return prob;
	
}

double ParticleFilter::multivariate_gaussian(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) 
{
return exp(-(
	  (pow(x - mu_x, 2) / (2 * pow(sig_x, 2)) +
	   pow(y - mu_y, 2) / (2 * pow(sig_y, 2))
	  ))) / (2 * M_PI * sig_x * sig_y);
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

