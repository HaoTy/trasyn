OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
h q[3];
cx q[0],q[1];
rz(2.109590535931358) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(2.06268683722817) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(2.2553519024204594) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(1.9322636440788739) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(2.5989853429772287) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(2.0655512862831658) q[2];
cx q[3],q[2];
rx(3.241422747964908) q[0];
rx(3.241422747964908) q[1];
rx(3.241422747964908) q[2];
rx(3.241422747964908) q[3];
cx q[0],q[1];
rz(4.922446727390318) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.813003233853751) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(5.262561336947689) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(4.5086781957201705) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(6.064383906816093) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(4.8196870417474535) q[2];
cx q[3],q[2];
rx(5.386545334086251) q[0];
rx(5.386545334086251) q[1];
rx(5.386545334086251) q[2];
rx(5.386545334086251) q[3];
cx q[0],q[1];
rz(4.265375239619999) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(4.1705407816113755) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(4.560089741285748) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(3.906838489978945) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(5.254881239394728) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(4.17633240319207) q[2];
cx q[3],q[2];
rx(4.557087421578892) q[0];
rx(4.557087421578892) q[1];
rx(4.557087421578892) q[2];
rx(4.557087421578892) q[3];
cx q[0],q[1];
rz(1.250368216484114) q[1];
cx q[0],q[1];
cx q[0],q[2];
rz(1.2225680850865197) q[2];
cx q[0],q[2];
cx q[3],q[0];
rz(1.3367619392208354) q[0];
cx q[3],q[0];
cx q[1],q[2];
rz(1.1452653987932992) q[2];
cx q[1],q[2];
cx q[3],q[1];
rz(1.5404357445754722) q[1];
cx q[3],q[1];
cx q[3],q[2];
rz(1.2242658629230718) q[2];
cx q[3],q[2];
rx(1.201287294680052) q[0];
rx(1.201287294680052) q[1];
rx(1.201287294680052) q[2];
rx(1.201287294680052) q[3];
