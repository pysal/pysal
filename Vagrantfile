# -*- mode: ruby -*-
# vi: set ft=ruby :

MACHINE_NAME = "PYVM"

Vagrant.configure("2") do |config|
  #config.vm.box = "1404amd64"
  config.vm.box = "base"
  config.vm.box_url = "http://cloud-images.ubuntu.com/vagrant/trusty/current/trusty-server-cloudimg-amd64-vagrant-disk1.box"

  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true

  config.vm.network :forwarded_port, guest: 8888, host: 8888, auto_correct: true

$apt_get_install =<<EOF
apt-get update -qq
apt-get install -y build-essential
apt-get install -y git-core
apt-get install -y python-dev
apt-get install -y python-pip
apt-get install -y python-numpy
apt-get install -y python-scipy
apt-get install -y python-matplotlib
apt-get install -y python-pandas
pip install ipython[notebook]
pip install sympy[all]
pip install clusterpy[all]
pip install -U pyzmq
pip install -U jinja2
pip install -U tornado
pip install -U pygments
EOF

$guest_additions = <<EOF
mkdir /media/VBoxGuestAdditions
mount -o loop,ro /vagrant/VBoxGuestAdditions_4.3.12.iso /media/VBoxGuestAdditions
/usr/bin/yes | bash /media/VBoxGuestAdditions/VBoxLinuxAdditions.run --nox11
umount /media/VBoxGuestAdditions
rmdir /media/VBoxGuestAdditions
EOF

$gitconfig =<<EOF
git config --global user.name \"Sergio Rey\"
git config --global user.email sjsrey@gmail.com
git config --global color.branch auto
git config --global color.status auto
git config --global color.ui 1
git config --global alias.ci commit
git config --global alias.co checkout
git config --global alias.st status \-sb
git config --global credential.helper store
EOF

$ipython_notebook = <<CONF_SCRIPT
ipython profile create
echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/vagrant/.ipython/profile_default/ipython_notebook_config.py
echo "c.IPKernelApp.pylab = 'inline'" >> /home/vagrant/.ipython/profile_default/ipython_notebook_config.py
mkdir -p /home/vagrant/.config/matplotlib
echo "backend: Qt4Agg" >> /home/vagrant/.config/matplotlib/matplotlibrc
CONF_SCRIPT

  _bashrc =  'echo -e "force_color_prompt=yes" >> /home/vagrant/.bashrc;'
  _bashrc << 'echo -e "red_color=\'\e[1;31m\'" >> /home/vagrant/.bashrc;'
  _bashrc << 'echo -e "end_color=\'\e[0m\'"    >> /home/vagrant/.bashrc;'
  _bashrc << "echo -e 'PS1=\"[\${red_color}#{MACHINE_NAME}\${end_color}]$ \"' >> /home/vagrant/.bashrc;"
  _bashrc << 'echo -e alias notebook=\"ipython notebook\" >> /home/vagrant/.bashrc;'
  _bashrc << 'echo -e export EDITOR=\"vi\" >> /home/vagrant/.bashrc;'
  _bashrc << 'echo -e export PYTHONPATH=\"/vagrant\" >> /home/vagrant/.bashrc;'

  _bash_login =  'echo -e "cd /vagrant" >> /home/vagrant/.bash_login;'
  _bash_login << 'echo -e "source ~/.bashrc" >> /home/vagrant/.bash_login;'

  config.vm.provision :shell, :inline => $apt_get_install
#  config.vm.provision :shell, :inline => $guest_additions
  config.vm.provision :shell, :inline => $gitconfig, :privileged => false
  config.vm.provision :shell, :inline => _bashrc
  config.vm.provision :shell, :inline => _bash_login
  config.vm.provision :shell, :inline => "touch ~/.hushlogin", :privileged => false
  config.vm.provision :shell, :inline => $ipython_notebook, :privileged => false

end
