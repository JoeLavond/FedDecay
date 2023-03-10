 # remove secure connections
  echo "Run '. ./dependencies.sh' to run in current shell"
  export http_proxy=http://152.2.41.28:3128
  export HTTP_PROXY=http://152.2.41.28:3128
  export https_proxy=http://152.2.41.28:3128
  export HTTPS_PROXY=http://152.2.41.28:3128
  echo "Verify echo \$http_proxy=$http_proxy is not secure"
