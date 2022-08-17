#ifndef GAZEBO_PLUGINS_BALLAERODYNAMICSPLUGIN_HH_
#define GAZEBO_PLUGINS_BALLAERODYNAMICSPLUGIN_HH_

#include <string>
#include <vector>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/transport/TransportTypes.hh"

namespace gazebo
{
	class BallAerodynamicsPlugin : public ModelPlugin
	{
		public: BallAerodynamicsPlugin();

		public: ~BallAerodynamicsPlugin();

		public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

		public: virtual void Init();

		protected: virtual void OnUpdate();

		protected: event::ConnectionPtr updateConnection;

		protected: physics::WorldPtr world;

		protected: physics::ModelPtr model;

		protected: double c_lin;

		protected: double c_ang;

		protected: double vel_pow;

		protected: std::string linkName;

		protected: physics::LinkPtr link;

		protected: sdf::ElementPtr sdf;
	};
}
#endif