const Sequelize = require('sequelize');

const db = require("../services/db")


const CreateFormObject = db.define('create_form', {
    name : {
        type: Sequelize.STRING,
        allowNull: false
    }
    // to do fill this out based on Google Docs
})