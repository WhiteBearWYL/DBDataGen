CREATE TABLE `authors` (
	`id` INT(11) NOT NULL AUTO_INCREMENT ,
	`first_name` VARCHAR(50) NOT NULL SKEW(0.3) ,
	`last_name` VARCHAR(50) NOT NULL SKEW(0.8),
	`email` VARCHAR(100) UNIQUE RULER("$@email.$.com"),
    `sex` CHAR(30) NOT NULL HISTOGRAM({'Male':0.48,'Female':0.48,'Other':0.04}),
	`birthdate` DATE NOT NULL,
	PRIMARY KEY (`id`)
) SIZE = 100;

CREATE TABLE `books` (
    `id` INT(11) NOT NULL AUTO_INCREMENT,
    `title` VARCHAR(150) NOT NULL SET('Fiction','Non-Fiction','Science','Biography','History','Children','Fantasy','Mystery','Romance','Horror'),
    `author_id` INT(11) NOT NULL,
    `published_date` DATE NOT NULL,
    `isbn` VARCHAR(20) NOT NULL UNIQUE,
    `pages` INT(11) NOT NULL DISTRI(NORMAL(80,15)),
    PRIMARY KEY (`id`),
    FOREIGN KEY (`author_id`) REFERENCES `authors`(`id`) ON DELETE CASCADE
) SIZE = 100;

CREATE TABLE `stocks` (
    `book_id` INT(11) NOT NULL,
    `author_id` INT(11) NOT NULL,
    `quantity` DECIMAL(5,2) NOT NULL RANGE(1,1000) SKEW(0.2), 
    PRIMARY KEY (`book_id`, `author_id`, `quantity`),
    FOREIGN KEY (`book_id`) REFERENCES `books`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (`author_id`) REFERENCES `authors`(`id`) ON DELETE CASCADE ON UPDATE CASCADE
) SIZE = 60;

CREATE TABLE `recall` (
    `book_id` INT(11) NOT NULL,
    `author_id` INT(11) NOT NULL,
    `number` INT(11) NOT NULL DISTRI(POISSON(5)),
    `recall_date` DATE NOT NULL,
    FOREIGN KEY (`book_id`) REFERENCES `books`(`id`) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (`author_id`) REFERENCES `authors`(`id`) ON DELETE CASCADE ON UPDATE CASCADE
) SIZE = 10000;