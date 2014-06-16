# -*- mode: ruby -*-
# vi: set ft=ruby :

MACHINE_NAME = "PySALvm"
# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  # All Vagrant configuration is done here. The most common configuration
  # options are documented and commented below. For a complete reference,
  # please see the online documentation at vagrantup.com.

  # Every Vagrant virtual environment requires a box to build off of.
  config.vm.box = "ubuntu/trusty32"
  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true
  config.vm.network :forwarded_port, host: 8888, guest: 8888

$requirements = <<END
apt-get update -qq
apt-get install -y build-essential
apt-get install -y git-core
apt-get install -y python-dev
apt-get install -y python-pip
apt-get install -y python-numpy
apt-get install -y python-scipy
apt-get install -y python-matplotlib
apt-get install -y python-pandas
apt-get install -y python-networkx
apt-get install -y python-qt4
apt-get install -y qt4-dev-tools
apt-get install -y python-sip
apt-get install -y python-sip-dev
apt-get install -y python-tk
pip install ipython[notebook]
pip install -U pyzmq
pip install -U jinja2
pip install -U tornado
pip install -U pygments
pip install -U pysal
pip install -U clusterpy
END

$ipython_notebook = <<CONF_SCRIPT
ipython profile create
echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/vagrant/.ipython/profile_default/ipython_notebook_config.py
echo "c.IPKernelApp.pylab = 'inline'" >> /home/vagrant/.ipython/profile_default/ipython_notebook_config.py
mkdir -p /home/vagrant/.config/matplotlib
echo "backend: Qt4AGG" >> /home/vagrant/.config/matplotlib/matplotlibrc
CONF_SCRIPT

_bashrc = 'echo -e "force_color_prompt=yes" >> /home/vagrant/.bashrc;'
_bashrc << 'echo -e "red_color=\'\e[1;31m\'" >> /home/vagrant/.bashrc;'
_bashrc << 'echo -e "end_color=\'\e[0m\'" >> /home/vagrant/.bashrc;'
_bashrc << "echo -e 'PS1=\"[\${red_color}#{MACHINE_NAME}\${end_color}]$ \"' >> /home/vagrant/.bashrc;"
_bashrc << 'echo -e alias netebook=\"ipython notebook\" >> /home/vagrant/.bashrc;'
_bashrc << 'echo -e export EDITOR=\"vi\" >> /home/vagrant/.bashrc;'
_bashrc << 'echo -e export PYTHONPATH=\"/vagrant\" >> /home/vagrant/.bashrc;'

_bash_login = 'echo -e "cd /vagrant" >> /home/vagrant/.bash_login;'
_bash_login << 'echo -e "source ~/.bashrc" >> /home/vagrant/.bash_login;'



  config.vm.provision :shell, :inline => $requirements
  config.vm.provision :shell, :inline => $ipython_notebook, :privileged => false
  config.vm.provision :shell, :inline => _bashrc
  config.vm.provision :shell, :inline => _bash_login
  config.vm.provision :shell, :inline => "touch ~/.huslogin", :privileged => false


end
