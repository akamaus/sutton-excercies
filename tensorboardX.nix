{lib, fetchFromGitHub, buildPythonPackage, numpy, protobuf, six, pytest}:

buildPythonPackage rec {
  pname = "tensorboardX";
  rev = "v1.2";
  name = "${pname}-${rev}";

  src = fetchFromGitHub {
    owner = "lanpa";
    repo = "tensorboard-pytorch";
    inherit rev;
    sha256 = "1mxv800sxzjhzj91iy6gl6l736v1m5l1dhm0cxypph8kk53vxdbw";
  };

  buildInputs = [ pytest ];

  propagatedBuildInputs = [ numpy protobuf six ];

  doCheck = false;

  meta = {
    description = "tensorboard for pytorch";
    homepage = https://github.com/lanpa/tensorboard-pytorch;
    maintainers = with lib.maintainers; [ lebastr ];
  };
}
