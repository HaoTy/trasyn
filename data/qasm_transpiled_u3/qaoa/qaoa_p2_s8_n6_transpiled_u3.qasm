OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[2];
cx q[0],q[2];
u3(0,0,3.094428596285997) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(0,0,3.7147634021904468) q[2];
cx q[1],q[2];
u3(pi/2,0,pi) q[3];
cx q[0],q[3];
u3(0,0,3.159179410984711) q[3];
cx q[0],q[3];
cx q[1],q[3];
u3(0,0,3.893248963260219) q[3];
cx q[1],q[3];
u3(pi/2,0,pi) q[4];
cx q[4],q[0];
u3(1.4658780821292015,pi/2,2.0491189676343424) q[0];
cx q[4],q[0];
cx q[4],q[1];
u3(1.4658780821292015,pi/2,2.022311564869259) q[1];
cx q[4],q[1];
u3(pi/2,0,pi) q[5];
cx q[5],q[2];
u3(1.4658780821292015,pi/2,2.0679310731842726) q[2];
cx q[5],q[2];
cx q[0],q[2];
u3(0,2.0552658417255287,2.0552658417255287) q[2];
cx q[0],q[2];
cx q[1],q[2];
u3(0,0,4.934562936612999) q[2];
cx q[1],q[2];
cx q[5],q[3];
u3(1.4658780821292015,pi/2,1.8586171176491426) q[3];
cx q[5],q[3];
cx q[0],q[3];
u3(0,2.0982722106021257,2.0982722106021257) q[3];
cx q[0],q[3];
cx q[1],q[3];
u3(0,0,5.171656968996363) q[3];
cx q[1],q[3];
cx q[5],q[4];
u3(1.4658780821292015,pi/2,1.9359047332952395) q[4];
cx q[5],q[4];
u3(4.817307225050385,-pi/2,pi/2) q[5];
cx q[4],q[0];
u3(0.8080748376980628,-pi/2,0.09618101764967779) q[0];
cx q[4],q[0];
cx q[4],q[1];
u3(0.8080748376980627,-pi/2,0.06057099305606961) q[1];
cx q[4],q[1];
cx q[5],q[2];
u3(0.8080748376980628,-pi/2,0.12117036656799662) q[2];
cx q[5],q[2];
cx q[5],q[3];
u3(0.8080748376980625,-pi/2,-0.15687503640180545) q[3];
cx q[5],q[3];
cx q[5],q[4];
u3(0.8080748376980627,-pi/2,-0.05420884684003191) q[4];
cx q[5],q[4];
u3(0.8080748376980627,-pi/2,pi/2) q[5];
