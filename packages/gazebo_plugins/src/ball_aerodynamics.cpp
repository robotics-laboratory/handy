#include "ball_aerodynamics.hh"

#include <algorithm>
#include <string>

#include "gazebo/common/Assert.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/sensors/SensorManager.hh"
#include "gazebo/transport/transport.hh"
#include <ignition/math/Vector3.hh>

using namespace gazebo;

GZ_REGISTER_MODEL_PLUGIN(BallAerodynamicsPlugin)

/////////////////////////////////////////////////
BallAerodynamicsPlugin::BallAerodynamicsPlugin() 
{
}

BallAerodynamicsPlugin::~BallAerodynamicsPlugin() 
{
}

/////////////////////////////////////////////////
void BallAerodynamicsPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
	GZ_ASSERT(_model, "BallAerodynamicsPlugin _model pointer is NULL");
	GZ_ASSERT(_sdf, "BallAerodynamicsPlugin _sdf pointer is NULL");
	this->model = _model;
	this->sdf = _sdf;

	this->world = this->model->GetWorld();
	GZ_ASSERT(this->world, "BallAerodynamicsPlugin world pointer is NULL");

	GZ_ASSERT(_sdf, "BallAerodynamicsPlugin _sdf pointer is NULL");

	if (_sdf->HasElement("c_linear_force"))
		this->c_lin = _sdf->Get<double>("c_linear_force");

	if (_sdf->HasElement("c_angular_force"))
		this->c_ang = _sdf->Get<double>("c_angular_force");

	if (_sdf->HasElement("velocity_pow"))
		this->vel_pow = _sdf->Get<double>("velocity_pow");

	if (_sdf->HasElement("link_name"))
  	{
    	sdf::ElementPtr elem = _sdf->GetElement("link_name");
    	this->linkName = elem->Get<std::string>();
    	this->link = this->model->GetLink(this->linkName);
  	}
}

/////////////////////////////////////////////////
void BallAerodynamicsPlugin::Init()
{
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(
					boost::bind(&BallAerodynamicsPlugin::OnUpdate, this));
}

/////////////////////////////////////////////////
void BallAerodynamicsPlugin::OnUpdate()
{
	const ignition::math::Vector3 cp = ignition::math::Vector3(0.0, 0.0, 0.0);
	ignition::math::Vector3 lin_vel = this->link->WorldLinearVel();
	ignition::math::Vector3 ang_vel = this->link->WorldAngularVel();
	ignition::math::Vector3 pow_lin_vel = lin_vel;

	if (lin_vel.Length() <= 0.01 && ang_vel.Length() <= 0.01)
		return;

	if (this->vel_pow == 2){
		pow_lin_vel.X() *= pow_lin_vel.X();
		pow_lin_vel.Y() *= pow_lin_vel.Y();
		pow_lin_vel.Z() *= pow_lin_vel.Z();
	}

	ignition::math::Vector3 force = - (c_lin * pow_lin_vel + c_ang * ang_vel.Cross(lin_vel));

	this->link->AddForceAtRelativePosition(force, cp);
}













