OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[3];
cx q[1],q[3];
u3(0,0,3.012827316422543) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,3.4480644072360995) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[5];
cx q[0],q[5];
u3(0,0,2.9900670145315553) q[5];
cx q[0],q[5];
cx q[2],q[5];
u3(0,0,3.2964804346387013) q[5];
cx q[2],q[5];
cx q[4],q[5];
u3(3.114697321836836,pi/2,1.6057699627353044) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[6];
cx q[6],q[0];
u3(3.114697321836836,pi/2,2.0991395258258176) q[0];
cx q[6],q[0];
cx q[2],q[6];
u3(0,0,3.35728281666634) q[6];
cx q[2],q[6];
cx q[3],q[6];
u3(3.114697321836836,pi/2,1.5823820687806522) q[6];
cx q[3],q[6];
u3(pi/2,0,pi) q[7];
cx q[1],q[7];
u3(0,0,3.1399131448586135) q[7];
cx q[1],q[7];
u3(pi/2,0,pi) q[8];
cx q[8],q[1];
u3(3.114697321836836,pi/2,1.459743119094619) q[1];
cx q[8],q[1];
u3(pi/2,0,pi) q[9];
cx q[7],q[9];
u3(0,0,2.7215337096029835) q[9];
cx q[7],q[9];
cx q[8],q[9];
u3(0,0,2.7485794101577503) q[9];
cx q[8],q[9];
u3(pi/2,0,pi) q[10];
cx q[10],q[2];
u3(3.114697321836836,pi/2,1.7181436226596585) q[2];
cx q[10],q[2];
cx q[10],q[7];
u3(3.114697321836836,pi/2,1.7955538931292727) q[7];
cx q[10],q[7];
cx q[10],q[9];
u3(3.114697321836836,pi/2,1.462830906846449) q[9];
cx q[10],q[9];
u3(3.168487985342751,-pi/2,pi/2) q[10];
u3(pi/2,0,pi) q[11];
cx q[11],q[3];
u3(3.114697321836836,pi/2,1.4066413141207317) q[3];
cx q[11],q[3];
cx q[11],q[4];
u3(3.114697321836836,pi/2,1.5134776107206935) q[4];
cx q[11],q[4];
cx q[11],q[8];
u3(3.114697321836836,pi/2,1.7309200278875618) q[8];
cx q[11],q[8];
u3(3.168487985342751,-pi/2,pi/2) q[11];
