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
cx q[0],q[2];
rz(1.8458790782214876) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(2.333397787402257) q[3];
cx q[0],q[3];
cx q[7],q[0];
rz(2.2570610655754244) q[0];
cx q[7],q[0];
cx q[1],q[4];
rz(1.9218886401363673) q[4];
cx q[1],q[4];
cx q[1],q[5];
rz(1.9458599653214486) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(2.3611095826769444) q[1];
cx q[6],q[1];
cx q[2],q[3];
rz(1.8456130642326516) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(2.3503440471034933) q[2];
cx q[4],q[2];
cx q[7],q[3];
rz(2.3617042343463743) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(1.8404764947891077) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(1.9596305553423448) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(2.040974547968815) q[5];
cx q[7],q[5];
rx(5.143007809374287) q[0];
rx(5.143007809374287) q[1];
rx(5.143007809374287) q[2];
rx(5.143007809374287) q[3];
rx(5.143007809374287) q[4];
rx(5.143007809374287) q[5];
rx(5.143007809374287) q[6];
rx(5.143007809374287) q[7];
cx q[0],q[2];
rz(4.112147869257244) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(5.1982152313253644) q[3];
cx q[0],q[3];
cx q[7],q[0];
rz(5.028156481697656) q[0];
cx q[7],q[0];
cx q[1],q[4];
rz(4.281477790030063) q[4];
cx q[1],q[4];
cx q[1],q[5];
rz(4.334879789622627) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(5.259950044421515) q[1];
cx q[6],q[1];
cx q[2],q[3];
rz(4.111555257926258) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(5.235967176479573) q[2];
cx q[4],q[2];
cx q[7],q[3];
rz(5.261274776699077) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(4.100112291080937) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(4.365557152555525) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(4.546770824622336) q[5];
cx q[7],q[5];
rx(0.9465761035159393) q[0];
rx(0.9465761035159393) q[1];
rx(0.9465761035159393) q[2];
rx(0.9465761035159393) q[3];
rx(0.9465761035159393) q[4];
rx(0.9465761035159393) q[5];
rx(0.9465761035159393) q[6];
rx(0.9465761035159393) q[7];
cx q[0],q[2];
rz(1.1620198702765123) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(1.4689231955720359) q[3];
cx q[0],q[3];
cx q[7],q[0];
rz(1.4208676167201333) q[0];
cx q[7],q[0];
cx q[1],q[4];
rz(1.209869495053236) q[4];
cx q[1],q[4];
cx q[1],q[5];
rz(1.2249599506039663) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(1.4863683560541885) q[1];
cx q[6],q[1];
cx q[2],q[3];
rz(1.1618524088515219) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(1.4795912240101043) q[2];
cx q[4],q[2];
cx q[7],q[3];
rz(1.486742701840932) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(1.1586188298869649) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(1.2336288278985346) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(1.2848365894874783) q[5];
cx q[7],q[5];
rx(4.692702001337668) q[0];
rx(4.692702001337668) q[1];
rx(4.692702001337668) q[2];
rx(4.692702001337668) q[3];
rx(4.692702001337668) q[4];
rx(4.692702001337668) q[5];
rx(4.692702001337668) q[6];
rx(4.692702001337668) q[7];
cx q[0],q[2];
rz(0.489976379375284) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(0.6193849927673705) q[3];
cx q[0],q[3];
cx q[7],q[0];
rz(0.5991219153993081) q[0];
cx q[7],q[0];
cx q[1],q[4];
rz(0.5101526143109104) q[4];
cx q[1],q[4];
cx q[1],q[5];
rz(0.5165156438622992) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(0.6267409053376365) q[1];
cx q[6],q[1];
cx q[2],q[3];
rz(0.48990576772327993) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(0.6238832652004516) q[2];
cx q[4],q[2];
cx q[7],q[3];
rz(0.6268987516859776) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(0.488542299374756) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(0.5201709558054852) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(0.541763180053159) q[5];
cx q[7],q[5];
rx(5.055571785697805) q[0];
rx(5.055571785697805) q[1];
rx(5.055571785697805) q[2];
rx(5.055571785697805) q[3];
rx(5.055571785697805) q[4];
rx(5.055571785697805) q[5];
rx(5.055571785697805) q[6];
rx(5.055571785697805) q[7];
cx q[0],q[2];
rz(2.9250376889581675) q[2];
cx q[0],q[2];
cx q[0],q[3];
rz(3.697575075209901) q[3];
cx q[0],q[3];
cx q[7],q[0];
rz(3.576609519540816) q[0];
cx q[7],q[0];
cx q[1],q[4];
rz(3.0454848168038544) q[4];
cx q[1],q[4];
cx q[1],q[5];
rz(3.0834705280283368) q[5];
cx q[1],q[5];
cx q[6],q[1];
rz(3.74148805226431) q[1];
cx q[6],q[1];
cx q[2],q[3];
rz(2.9246161548759435) q[3];
cx q[2],q[3];
cx q[4],q[2];
rz(3.7244286480673106) q[2];
cx q[4],q[2];
cx q[7],q[3];
rz(3.7424303558883154) q[3];
cx q[7],q[3];
cx q[6],q[4];
rz(2.9164765863681335) q[4];
cx q[6],q[4];
cx q[5],q[6];
rz(3.1052918354398287) q[6];
cx q[5],q[6];
cx q[7],q[5];
rz(3.234192068943755) q[5];
cx q[7],q[5];
rx(1.7118764904012307) q[0];
rx(1.7118764904012307) q[1];
rx(1.7118764904012307) q[2];
rx(1.7118764904012307) q[3];
rx(1.7118764904012307) q[4];
rx(1.7118764904012307) q[5];
rx(1.7118764904012307) q[6];
rx(1.7118764904012307) q[7];
