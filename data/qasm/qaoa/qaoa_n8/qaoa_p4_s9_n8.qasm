OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
cx q[0],q[1];
rz(1.9095748340431697) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.9662137004106328) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(1.6671441337349868) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(2.0978214734510128) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.9521332163860858) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(1.7030724754149171) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(1.8753563204801738) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(1.5290962399409316) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(1.8636532928182377) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(1.8058898073877965) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.9074854174647664) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(1.6552473659535438) q[5];
cx q[6],q[5];
rx(5.964919674522166) q[0];
rx(5.964919674522166) q[1];
rx(5.964919674522166) q[2];
rx(5.964919674522166) q[3];
rx(5.964919674522166) q[4];
rx(5.964919674522166) q[5];
rx(5.964919674522166) q[6];
rx(5.964919674522166) q[7];
cx q[0],q[1];
rz(1.789300192091232) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.8423716573537658) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(1.5621389984595624) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(1.9656901099138333) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.8291780331392513) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(1.5958043922023695) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(1.7572369328778141) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(1.4327860563910406) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(1.7462710208484782) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(1.6921457706964798) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(1.7873423774937385) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(1.550991549039353) q[5];
cx q[6],q[5];
rx(5.286273321009105) q[0];
rx(5.286273321009105) q[1];
rx(5.286273321009105) q[2];
rx(5.286273321009105) q[3];
rx(5.286273321009105) q[4];
rx(5.286273321009105) q[5];
rx(5.286273321009105) q[6];
rx(5.286273321009105) q[7];
cx q[0],q[1];
rz(3.94290881528622) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.059857301151655) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(3.4423355314850066) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(4.331602319587194) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(4.030783779866116) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(3.5165207231847053) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(3.872254093424464) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(3.1572928886576497) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(3.8480895673157463) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(3.7288189569971597) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(3.938594567476225) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(3.417770968880489) q[5];
cx q[6],q[5];
rx(5.098822932901669) q[0];
rx(5.098822932901669) q[1];
rx(5.098822932901669) q[2];
rx(5.098822932901669) q[3];
rx(5.098822932901669) q[4];
rx(5.098822932901669) q[5];
rx(5.098822932901669) q[6];
rx(5.098822932901669) q[7];
cx q[0],q[1];
rz(0.14540070504665187) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(0.14971335671971325) q[2];
cx q[0],q[2];
cx q[5],q[0];
rz(0.1269413107766044) q[0];
cx q[5],q[0];
cx q[1],q[2];
rz(0.15973436383006273) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(0.14864122680468267) q[1];
cx q[3],q[1];
cx q[7],q[2];
rz(0.12967700152738687) q[2];
cx q[7],q[2];
cx q[3],q[6];
rz(0.14279520569202664) q[6];
cx q[3],q[6];
cx q[7],q[3];
rz(0.11642993372553502) q[3];
cx q[7],q[3];
cx q[4],q[5];
rz(0.14190410237264373) q[5];
cx q[4],q[5];
cx q[4],q[6];
rz(0.1375058188606249) q[6];
cx q[4],q[6];
cx q[7],q[4];
rz(0.14524161065651855) q[4];
cx q[7],q[4];
cx q[6],q[5];
rz(0.12603545550853115) q[5];
cx q[6],q[5];
rx(5.382184603414291) q[0];
rx(5.382184603414291) q[1];
rx(5.382184603414291) q[2];
rx(5.382184603414291) q[3];
rx(5.382184603414291) q[4];
rx(5.382184603414291) q[5];
rx(5.382184603414291) q[6];
rx(5.382184603414291) q[7];
