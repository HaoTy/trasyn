OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(3.2470811860931303) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(2.875651537601063) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(3.5514650764159814) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(3.2727160496057772) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.278542412862729) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(3.025235124936421) q[2];
cx q[3],q[2];
rx(1.8313435406231167) q[0];
rx(1.8313435406231167) q[1];
rx(1.8313435406231167) q[2];
rx(1.8313435406231167) q[3];
cx q[0],q[1];
rz(3.9469909952004447) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(3.4954995190317892) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(4.3169849698918785) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(3.978151465065182) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(3.9852337035409082) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(3.6773259158496683) q[2];
cx q[3],q[2];
rx(4.957039015075064) q[0];
rx(4.957039015075064) q[1];
rx(4.957039015075064) q[2];
rx(4.957039015075064) q[3];
cx q[0],q[1];
rz(0.8325211639774972) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(0.7372900854868408) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(0.9105623388495291) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(0.8390937025704208) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(0.8405875274680574) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(0.7756419144382437) q[2];
cx q[3],q[2];
rx(0.4889143883957278) q[0];
rx(0.4889143883957278) q[1];
rx(0.4889143883957278) q[2];
rx(0.4889143883957278) q[3];
cx q[0],q[1];
rz(0.8831432611498115) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(0.7821215828308622) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(0.9659297903848951) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(0.8901154480660644) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(0.8917001061489683) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(0.8228054245837635) q[2];
cx q[3],q[2];
rx(5.9341686118389445) q[0];
rx(5.9341686118389445) q[1];
rx(5.9341686118389445) q[2];
rx(5.9341686118389445) q[3];
