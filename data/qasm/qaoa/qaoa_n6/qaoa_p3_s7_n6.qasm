OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[0],q[1];
rz(3.858132265350553) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.273443585863638) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(4.223380982572884) q[0];
cx q[5],q[0];
cx q[1],q[4];
rz(4.043639973106589) q[4];
cx q[1],q[4];
cx q[5],q[1];
rz(3.9571705302657754) q[1];
cx q[5],q[1];
cx q[2],q[3];
rz(3.849618587047907) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(4.312396001016232) q[2];
cx q[4],q[2];
cx q[3],q[4];
rz(3.761554570966999) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(3.38595367750364) q[3];
cx q[5],q[3];
rx(5.513943923325995) q[0];
rx(5.513943923325995) q[1];
rx(5.513943923325995) q[2];
rx(5.513943923325995) q[3];
rx(5.513943923325995) q[4];
rx(5.513943923325995) q[5];
cx q[0],q[1];
rz(4.718612700043368) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.00352072318812) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(5.165322951850468) q[0];
cx q[5],q[0];
cx q[1],q[4];
rz(4.945494249345057) q[4];
cx q[1],q[4];
cx q[5],q[1];
rz(4.839739499872448) q[1];
cx q[5],q[1];
cx q[2],q[3];
rz(4.708200213430679) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(5.274191017441064) q[2];
cx q[4],q[2];
cx q[3],q[4];
rz(4.600495252554113) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(4.141123975431034) q[3];
cx q[5],q[3];
rx(2.2497125453285247) q[0];
rx(2.2497125453285247) q[1];
rx(2.2497125453285247) q[2];
rx(2.2497125453285247) q[3];
rx(2.2497125453285247) q[4];
rx(2.2497125453285247) q[5];
cx q[0],q[1];
rz(1.8151464852170467) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.5400663354128927) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(1.9869860904193608) q[0];
cx q[5],q[0];
cx q[1],q[4];
rz(1.9024228253099245) q[4];
cx q[1],q[4];
cx q[5],q[1];
rz(1.8617413000390644) q[1];
cx q[5],q[1];
cx q[2],q[3];
rz(1.8111410307161466) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(2.0288652399005915) q[2];
cx q[4],q[2];
cx q[3],q[4];
rz(1.769709302027386) q[4];
cx q[3],q[4];
cx q[5],q[3];
rz(1.592999278958114) q[3];
cx q[5],q[3];
rx(2.692547461793544) q[0];
rx(2.692547461793544) q[1];
rx(2.692547461793544) q[2];
rx(2.692547461793544) q[3];
rx(2.692547461793544) q[4];
rx(2.692547461793544) q[5];
