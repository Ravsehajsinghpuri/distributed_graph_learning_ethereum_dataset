for server in $(cat servers.txt) ; do ssh ${server} 'bash -s' < ./environment_setup.sh ; done