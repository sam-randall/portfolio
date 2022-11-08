
const Sequelize = require('sequelize');

const db = require("../services/db")




const Patient = db.define('patient', {

    nin: {
        type: Sequelize.INTEGER,
        allowNull: false
    },
    patient_name: {
        type : Sequelize.STRING,
        allowNull: false
    },
    outpatient_id_number : {
        type: Sequelize.INTEGER,
        allowNull: true
    },
    birth_date : {
        type: Sequelize.DATE
    },
    sex : {
        type: Sequelize.SMALLINT
    },
    ward : {
        type: Sequelize.STRING
    },
    phone_number: {
        type: Sequelize.STRING
    },
    address : {
        type: Sequelize.STRING
    }
});


module.exports = Patient;
