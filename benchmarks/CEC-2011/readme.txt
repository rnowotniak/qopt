MATLAB Compiler

1. Prerequisites for Deployment 

. Verify the MATLAB Compiler Runtime (MCR) is installed and ensure you    
  have installed version 7.15.   

. If the MCR is not installed, run MCRInstaller, located in:

  <matlabroot>*/toolbox/compiler/deploy/glnx86/MCRInstaller.bin

For more information on the MCR Installer, see the MATLAB Compiler 
documentation.    



2. Files to Deploy and Package

Files to package for Shared Libraries
=====================================
-CEC2011.so
-CEC2011.h
-MCRInstaller.bin 
   - include when building component by clicking "Add MCR" link
     in deploytool
-This readme file

3. Definitions

For a complete list of product terminology, go to 
http://www.mathworks.com/help and select MATLAB Compiler.



* NOTE: <matlabroot> is the directory where MATLAB is installed on the target machine.


4. Appendix 

A. On the target machine, add the MCR directory to the environment variable LD_LIBRARY_PATH.

        NOTE: <mcr_root> is the directory where MCR is installed
              on the target machine.         


        . Add the MCR directory to the environment variable by issuing 
          the following commands:

            setenv LD_LIBRARY_PATH
                <mcr_root>/v715/runtime/glnx86:
                <mcr_root>/v715/sys/os/glnx86:
                <mcr_root>/v715/sys/java/jre/glnx86/jre/lib/i386/native_threads:
                <mcr_root>/v715/sys/java/jre/glnx86/jre/lib/i386/server:
                <mcr_root>/v715/sys/java/jre/glnx86/jre/lib/i386
            setenv XAPPLRESDIR <mcr_root>/v715/X11/app-defaults


        NOTE: To make these changes persistent after logout on Linux 
              or Mac machines, modify the .cshrc file to include this  
              setenv command.
        NOTE: The environment variable syntax utilizes forward 
              slashes (/), delimited by colons (:).  
