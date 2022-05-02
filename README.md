# Readout-Script-Dev

This repo is for version-controlling general scratch development scripts for
detectors and readout. Many of the scripts here were originally in the
[sodetlib](https://github.com/simonsobs/sodetlib) scratch directory.

## Adding Readout Script Dev to Smurf-related dockers

For testing, it may be useful to have access to these scripts in the docker
containers on the smurf-server that are used for readout testing. In order
to add this, 

First, clone this repo onto the smurf-server to wherever git repos are stored.
This will likely be in the directory `/home/cryo/repos`, but may be somewhere
else if your smurf-server was set up much earlier.

Second, you'll need to mount the `readout-script-dev` directory into any docker
containers that you may want to use it from. To do this, you'll need to edit
the `docker-compose` file in the `$OCS_CONFIG_DIR`. We are already mounting
in the sodetlib repository, so a relatively safe way to figure out where to mount
this repo is to add the `readout-script-dev` mount below any existing
`sodetlib` mount. For instance, the `x-smurf-service` block may look something
like this:
```
x-smurf-service: &smurf-service                                                     
    user: cryo:smurf                                                                
    network_mode: host                                                              
    security_opt:                                                                   
        - "apparmor=docker-smurf"                                                   
    volumes:                                                                        
        - ${HOME}/.Xauthority:${HOME}/.Xauthority                                   
        - ${HOME}/.bash_history:${HOME}/.bash_history                               
        - ${HOME}/.ipython:${HOME}/.ipython                                         
        - /data:/data                                                               
        - ${OCS_CONFIG_DIR}:/config                                                 
        - /home/cryo/repos/sodetlib:/sodetlib
        - /home/cryo/repos/readout-script-dev:/readout-script-dev                   
        - /home/cryo/repos/pysmurf/python/pysmurf:/usr/local/src/pysmurf/python/pysmurf
        - /home/cryo/repos/pysmurf:/usr/local/src/pysmurf                       
```

Finally, you'll want to modify local scripts that you use that reference the
directory `/sodetlib/scratch` to instead reference the directory
`/readout-script-dev`.
